"""Chapter 3: the Instruct model's generative MMLU score with a larger token budget (256 instead of 48 new tokens).
Same 400 questions and prompt as the 'generate' run in ch3_mmlu.py; the answer is read with the strict rule of
ch3_mmlu_parse.py (a letter at the start, or "answer is X", or "Answer: X"), falling back to the last standalone letter."""
import json, re
from common import Log, save
from ch3_common import INSTRUCT, data, load, chat, generate, sample_rows
from ch3_mmlu_fmt import L, fmt, header

log = Log('ch3_mmlu_long')
test = sample_rows(data('mmlu_test'), 400, seed=0)
tok, model = load(INSTRUCT)
prompts = [chat(tok, header(r.subject) + fmt(r.question, list(r.choices))[:-len('Answer:')]
                + 'Answer with the letter of the correct option.') for r in test.itertuples()]
outs = generate(tok, model, prompts, max_new_tokens=256, batch=16)


def read(t):
    for p in [r'\A\s*\(?\**([ABCD])\b', r'answer is[:\s]*\(?\**([ABCD])\b', r'Answer:\s*\(?\**([ABCD])\b']:
        m = re.search(p, t)
        if m:
            return L.index(m.group(1))
    m = re.findall(r'\b([ABCD])\b', t)
    return L.index(m[-1]) if m else -1


preds = [read(o) for o in outs]
ok = [int(p == a) for p, a in zip(preds, test['answer'])]
stopped = sum(1 for o in outs if len(tok(o)['input_ids']) < 256)
log(f'instruct, generate with 256 new tokens: acc {sum(ok) / 400:.3f}   no letter found: {preds.count(-1)}   '
    f'replies that ended by themselves: {stopped} of 400')
save('ch3_mmlu_long', dict(correct=ok, pred=preds, text=outs))
