"""Chapter 3, experiment 9: optimising against a reward model (best-of-n), with a ground truth to check it.
Qwen2.5-0.5B-Instruct writes 16 sampled solutions (temperature 0.8) for each of 150 GSM8K test questions.
A reward model (Skywork-Reward-V2-Qwen3-0.6B) scores every solution. For n = 1, 2, 4, 8, 16 we keep the solution the
reward model likes best among the first n, and measure two things:
  proxy   the reward model's score of the kept solution (what the optimisation pushes up)
  gold    whether its final answer is right (what we actually want)
Baselines on the same samples: majority vote over the n answers, and the oracle (any of the n is right = pass@n)."""
import re
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import Log, save
from ch3_common import INSTRUCT, DEV, data, load, chat, generate, sample_rows

log = Log('ch3_bestofn')
NQ, NS = 150, 16
test = sample_rows(data('gsm8k_test'), NQ, seed=5)
INSTR = 'Please reason step by step, and put your final answer within \\boxed{}.'


def norm(s):
    s = s.strip().replace(',', '').replace('$', '').rstrip('.')
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return s


def extract(text):
    m = re.findall(r'\\boxed\{\s*(-?\$?\d[\d,]*(?:\.\d+)?)\s*\}', text)
    return norm(m[-1]) if m else None


refs = [norm(a.split('####')[-1]) for a in test['answer']]
tok, model = load(INSTRUCT)
prompts = [chat(tok, q + '\n' + INSTR) for q in test['question'] for _ in range(NS)]
outs = generate(tok, model, prompts, max_new_tokens=320, batch=48, do_sample=True, temperature=0.8, top_p=1.0, seed=0)
del model
samples = [outs[i * NS:(i + 1) * NS] for i in range(NQ)]
answers = [[extract(o) for o in s] for s in samples]
gold = np.array([[int(a == r) for a in ans] for ans, r in zip(answers, refs)])
log(f'{NQ} questions x {NS} samples; {gold.mean():.3f} of all samples are right; no boxed answer in {sum(a is None for x in answers for a in x)}')

name = 'Skywork/Skywork-Reward-V2-Qwen3-0.6B'
rtok = AutoTokenizer.from_pretrained(name)
rm = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32, num_labels=1).to(DEV).eval()
scores = np.zeros((NQ, NS))
with torch.no_grad():
    for i, (q, s) in enumerate(zip(test['question'], samples)):
        for j, o in enumerate(s):
            conv = [{'role': 'user', 'content': q + '\n' + INSTR}, {'role': 'assistant', 'content': o}]
            ids = rtok.apply_chat_template(conv, tokenize=True, return_tensors='pt', return_dict=True)['input_ids'].to(DEV)
            scores[i, j] = rm(input_ids=ids).logits[0, 0].item()
log(f'mean reward of right samples {scores[gold == 1].mean():.2f}, of wrong samples {scores[gold == 0].mean():.2f}')

res = {'nq': NQ, 'ns': NS, 'rows': [], 'p_right': float(gold.mean())}
log(f'{"n":>3s} {"proxy (RM score of pick)":>25s} {"gold (best-of-n acc)":>20s} {"majority vote":>14s} {"oracle pass@n":>14s}')
for n in [1, 2, 4, 8, 16]:
    pick = scores[:, :n].argmax(1)
    proxy = scores[np.arange(NQ), pick].mean()
    acc = gold[np.arange(NQ), pick].mean()
    maj = np.mean([int(Counter([a for a in ans[:n] if a] or [None]).most_common(1)[0][0] == r) for ans, r in zip(answers, refs)])
    orc = gold[:, :n].max(1).mean()
    res['rows'].append(dict(n=n, proxy=float(proxy), gold=float(acc), majority=float(maj), oracle=float(orc)))
    log(f'{n:3d} {proxy:25.2f} {acc:20.3f} {maj:14.3f} {orc:14.3f}')
# the per-question correlation between reward and correctness
from_rm = [(scores[i][gold[i] == 1].mean() > scores[i][gold[i] == 0].mean()) for i in range(NQ) if 0 < gold[i].sum() < NS]
res['rm_ranks_right_higher'] = float(np.mean(from_rm))
res['mixed_questions'] = len(from_rm)
log(f'questions with both right and wrong samples: {len(from_rm)}; the RM gives the right ones a higher average score in {np.mean(from_rm):.2f} of them')
# the highest-scoring wrong answer, to read
i, j = max(((i, j) for i in range(NQ) for j in range(NS) if gold[i, j] == 0), key=lambda t: scores[t])
res['top_wrong'] = dict(q=test['question'][i], text=samples[i][j], score=scores[i, j], ref=refs[i], got=answers[i][j])
log(f'\nthe wrong solution with the highest reward ({scores[i, j]:.2f}); reference {refs[i]}, it says {answers[i][j]}:')
log('  ' + samples[i][j][-300:].replace('\n', ' | '))
save('ch3_bestofn', res)
