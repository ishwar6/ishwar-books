"""Chapter 3, experiment 3: GSM8K (grade-school maths word problems) on a random 300-question subset.
Base model: 4 solved examples from the GSM8K training set (few-shot), then the question; it writes a solution.
Instruct model: the question in the chat template with the usual instruction to reason step by step and box the answer.
Both: greedy decoding, up to 320 new tokens. The answer is pulled out of the text in two ways:
  strict    only the format we asked for ("#### 42" for base, "\\boxed{42}" for Instruct)
  flexible  the last number anywhere in the text
and compared with the reference number (exact match after removing commas, "$" and a trailing ".")."""
import re, time
from common import Log, save
from ch3_common import BASE, INSTRUCT, data, load, chat, generate, sample_rows

log = Log('ch3_gsm8k')
N = 300
test = sample_rows(data('gsm8k_test'), N, seed=0)
train = data('gsm8k_train')
SHOTS = train.iloc[[0, 1, 2, 3]]


def clean(sol):
    return re.sub(r'<<[^>]*>>', '', sol)            # drop the calculator notes, e.g. <<48/2=24>>


def ref(sol):
    return norm(sol.split('####')[-1])


def norm(s):
    s = s.strip().replace(',', '').replace('$', '').rstrip('.')
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


NUM = r'-?\$?\d[\d,]*(?:\.\d+)?'


def strict(text, kind):
    if kind == 'base':
        m = re.search(r'####\s*(' + NUM + ')', text)
    else:
        m = re.search(r'\\boxed\{\s*(' + NUM + r')\s*\}', text)
    return norm(m.group(1)) if m else None


def flexible(text):
    nums = re.findall(NUM, text)
    return norm(nums[-1]) if nums else None


FEW = ''.join(f'Question: {r.question}\nAnswer: {clean(r.answer)}\n\n' for r in SHOTS.itertuples())
INSTR = 'Please reason step by step, and put your final answer within \\boxed{}.'
R = {'n': N, 'ref': [ref(a) for a in test['answer']], 'models': {}}
log(f'GSM8K subset: {N} of {len(data("gsm8k_test"))} test questions (random, seed 0); base uses 4-shot, Instruct uses the chat template')
for name, label in [(BASE, 'base'), (INSTRUCT, 'instruct')]:
    tok, model = load(name)
    t0 = time.time()
    if label == 'base':
        prompts = [FEW + f'Question: {q}\nAnswer:' for q in test['question']]
    else:
        prompts = [chat(tok, q + '\n' + INSTR) for q in test['question']]
    outs = generate(tok, model, prompts, max_new_tokens=320, batch=20)
    if label == 'base':                               # a base model keeps writing new questions; cut there
        outs = [o.split('\n\nQuestion:')[0].split('\nQuestion:')[0] for o in outs]
    s = [strict(o, label) for o in outs]
    f = [flexible(o) for o in outs]
    sc = [int(a == b) for a, b in zip(s, R['ref'])]
    fc = [int(a == b) for a, b in zip(f, R['ref'])]
    R['models'][label] = dict(text=outs, strict=s, flexible=f, strict_correct=sc, flexible_correct=fc)
    log(f'{label:8s} strict {sum(sc) / N:.3f}  flexible {sum(fc) / N:.3f}  no strict answer found: {s.count(None)}  '
        f'mean length {sum(len(o) for o in outs) / N:.0f} chars  ({time.time() - t0:.0f}s)')
    del model
save('ch3_gsm8k', R)

# a few examples, to read
for label in ['base', 'instruct']:
    M = R['models'][label]
    disagree = [i for i in range(N) if M['strict_correct'][i] != M['flexible_correct'][i]]
    log(f'\n{label}: {len(disagree)} questions where strict and flexible extraction disagree')
    for i in disagree[:2]:
        log(f'  Q{i}: reference {R["ref"][i]}, strict {M["strict"][i]}, flexible {M["flexible"][i]}')
        log('  ...' + M['text'][i][-160:].replace('\n', ' | '))
