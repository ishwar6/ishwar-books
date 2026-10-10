"""Chapter 3, experiment 7: a small model as a judge, and its position and length biases.
Judges: Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct (both far weaker than the GPT-4 judge in MT-Bench).
Pairs where we know which answer is better:
  easy      100 pairs from RewardBench "alpacaeval-easy": a strong chat answer vs a weak one
  maths     100 pairs from RewardBench "math-prm": a correct solution vs one with a wrong step
  length    60 GSM8K reference solutions: the plain solution vs the same solution padded with polite filler
            (no new information; an unbiased judge should split 50/50)
Each pair is judged twice, with the answers swapped. The judge's verdict is read from its next-token probabilities:
after the prompt we write "My verdict: [[" and compare P("A") with P("B"), as in MT-Bench's "[[A]]" / "[[B]]" format."""
import random, re
import torch
from common import Log, save
from ch3_common import INSTRUCT, DEV, data, load, sample_rows

log = Log('ch3_judge')
JUDGES = [(INSTRUCT, 'Qwen2.5-0.5B-Instruct'), ('Qwen/Qwen2.5-1.5B-Instruct', 'Qwen2.5-1.5B-Instruct')]
CUT = 1500
SYSTEM = ('Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the '
          'user question displayed below. Choose the assistant that follows the user\'s instructions and answers the '
          'question better. Consider helpfulness, correctness and level of detail. Do not let the order in which the '
          'responses were presented or their length influence your decision. Output your final verdict as "[[A]]" if '
          'assistant A is better, or "[[B]]" if assistant B is better.')


def user_msg(q, a, b):
    return (f'[User Question]\n{q}\n\n[The Start of Assistant A\'s Answer]\n{a[:CUT]}\n[The End of Assistant A\'s Answer]\n\n'
            f'[The Start of Assistant B\'s Answer]\n{b[:CUT]}\n[The End of Assistant B\'s Answer]')


PAD_OPEN = ['Great question! Let me walk you through this carefully, step by step, so that everything is completely clear.',
            'What a nice problem! I will explain my reasoning in detail so that you can follow every single step.']
PAD_CLOSE = ['I hope this detailed explanation helps! Word problems like this one are a great way to practise careful '
             'reasoning, and breaking them into small steps always makes them easier. Let me know if you have any other questions!',
             'I hope this was helpful! Remember that reading the question slowly and writing down each quantity is the key to '
             'solving problems like this. Feel free to ask if anything is unclear!']


def gsm_solution(sol):
    body, ans = sol.split('####')
    return re.sub(r'<<[^>]*>>', '', body).strip() + f'\nThe answer is {ans.strip()}.'


rb = data('rewardbench')
sets = {}
for sub, key in [('alpacaeval-easy', 'easy'), ('math-prm', 'maths')]:
    d = sample_rows(rb[rb.subset == sub], 100, seed=0)
    sets[key] = [(r.prompt, r.chosen, r.rejected) for r in d.itertuples()]
g = sample_rows(data('gsm8k_test'), 60, seed=3)
rnd = random.Random(0)
sets['length'] = [(q, ' '.join([rnd.choice(PAD_OPEN), gsm_solution(s), rnd.choice(PAD_CLOSE)]), gsm_solution(s))
                  for q, s in zip(g['question'], g['answer'])]       # "better" = padded, only to measure the lean


@torch.no_grad()
def p_first(tok, model, q, a, b):
    """P(the judge says A) normalised over the two verdict tokens."""
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_msg(q, a, b)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + 'My verdict: [['
    x = tok(prompt, return_tensors='pt')['input_ids'].to(DEV)
    pr = torch.softmax(model(x).logits[0, -1].float(), -1)
    pa, pb = pr[tok.encode('A')[0]].item(), pr[tok.encode('B')[0]].item()
    return pa / (pa + pb), pa + pb


R = {'sets': {k: len(v) for k, v in sets.items()}, 'judges': {}}
for name, label in JUDGES:
    tok, model = load(name)
    R['judges'][label] = {}
    log(f'== judge: {label} ==')
    for key, pairs in sets.items():
        rows = []
        for q, good, bad in pairs:
            p1, m1 = p_first(tok, model, q, good, bad)        # good answer shown first (as A)
            p2, m2 = p_first(tok, model, q, bad, good)        # good answer shown second (as B)
            rows.append(dict(p_good_first=p1, p_good_second=1 - p2, mass=(m1 + m2) / 2))
        n = len(rows)
        acc1 = sum(r['p_good_first'] > 0.5 for r in rows) / n           # picks the good one when it is A
        acc2 = sum(r['p_good_second'] > 0.5 for r in rows) / n          # picks the good one when it is B
        cons = sum((r['p_good_first'] > 0.5) == (r['p_good_second'] > 0.5) for r in rows) / n
        first = sum(r['p_good_first'] > 0.5 and r['p_good_second'] <= 0.5 for r in rows) / n   # always says A
        second = sum(r['p_good_first'] <= 0.5 and r['p_good_second'] > 0.5 for r in rows) / n  # always says B
        both = sum(r['p_good_first'] > 0.5 and r['p_good_second'] > 0.5 for r in rows) / n
        avg = sum((r['p_good_first'] + r['p_good_second']) / 2 > 0.5 for r in rows) / n         # swap and average
        pA = sum((r['p_good_first'] + (1 - r['p_good_second'])) / 2 for r in rows) / n          # mean P("A") overall
        R['judges'][label][key] = dict(acc_good_first=acc1, acc_good_second=acc2, consistent=cons, always_first=first,
                                       always_second=second, consistent_right=both, swap_avg_acc=avg, mean_pA=pA,
                                       mass=sum(r['mass'] for r in rows) / n, rows=rows)
        what = 'prefers the padded answer' if key == 'length' else 'picks the better answer'
        log(f'{key:7s} n={n:3d}  {what}: when it is A {acc1:.2f}, when it is B {acc2:.2f} | consistent {cons:.2f} '
            f'(right both times {both:.2f}) | always "A" {first:.2f}, always "B" {second:.2f} | mean P("A") {pA:.2f} | '
            f'swap-and-average {avg:.2f}')
    del model
    if DEV == 'mps':
        torch.mps.empty_cache()
    log('')
save('ch3_judge', R)
