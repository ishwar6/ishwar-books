"""Chapter 3: GSM8K, Instruct model, with a larger token budget for the replies that had no \\boxed{} answer within 320
new tokens in ch3_gsm8k.py. Greedy decoding is deterministic, so rerunning those questions with 1,024 new tokens is the
same as letting the original replies continue. Also counts how many replies stopped by themselves."""
import json, re
from common import Log, save
from ch3_common import INSTRUCT, data, load, chat, generate, sample_rows

N = 300
INSTR = 'Please reason step by step, and put your final answer within \\boxed{}.'      # as in ch3_gsm8k.py
NUM = r'-?\$?\d[\d,]*(?:\.\d+)?'


def norm(s):                                               # as in ch3_gsm8k.py
    s = s.strip().replace(',', '').replace('$', '').rstrip('.')
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def strict(text):
    m = re.search(r'\\boxed\{\s*(' + NUM + r')\s*\}', text)
    return norm(m.group(1)) if m else None

log = Log('ch3_gsm8k_long')
G = json.load(open('results/ch3_gsm8k.json'))
test = sample_rows(data('gsm8k_test'), N, seed=0)
M = G['models']['instruct']
todo = [i for i in range(N) if M['strict'][i] is None]
tok, model = load(INSTRUCT)
cut = sum(1 for t in M['text'] if len(tok(t)['input_ids']) >= 319)
log(f'Instruct replies that hit the 320-token budget: {cut} of {N}; replies without a \\boxed{{}} answer: {len(todo)}')
outs = generate(tok, model, [chat(tok, test['question'][i] + '\n' + INSTR) for i in todo], max_new_tokens=1024, batch=12)
s = list(M['strict'])
for i, o in zip(todo, outs):
    s[i] = strict(o)
ok = [int(a == b) for a, b in zip(s, G['ref'])]
still = sum(v is None for v in s)
log(f'with up to 1,024 tokens for those {len(todo)}: strict accuracy {sum(ok) / N:.3f}   still no \\boxed{{}} answer: {still}')
save('ch3_gsm8k_long', dict(strict=s, strict_correct=ok, todo=todo, texts=outs, cut_at_320=cut))
