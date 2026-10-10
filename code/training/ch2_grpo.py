"""Chapter 2, experiment 4: GRPO's group-relative advantage (Shao et al. 2024, DeepSeekMath, section 4.1.2).

For one question, sample a group of G answers, score each with a checkable reward (1 if the final number is right, else 0),
and turn rewards into advantages:  A_i = (r_i - mean(r)) / std(r).  No value network is needed.
Part A: the formula on hand-picked reward lists. Part B: real groups of 4 answers from Qwen2.5-0.5B-Instruct,
a rule-based checker, the advantages, and one policy-gradient step to see which answers become more likely."""
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save

log = Log('ch2_grpo')
OUT = {}


def advantages(r, eps=1e-4):
    r = np.asarray(r, dtype=float)
    return (r - r.mean()) / (r.std() + eps)        # population std (divide by G); a small eps avoids 0/0


# ---------------------------------------------------------------- Part A
log('== A. group-relative advantages on hand-picked rewards ==')
cases = {'2 right, 2 wrong': [1, 0, 0, 1], '1 right, 3 wrong': [1, 0, 0, 0], '3 right, 1 wrong': [1, 1, 0, 1],
         'all right': [1, 1, 1, 1], 'all wrong': [0, 0, 0, 0], 'graded rewards': [0.9, 0.2, 0.5, 0.0]}
OUT['cases'] = {}
for name, r in cases.items():
    a = advantages(r)
    r_ = np.asarray(r, float)
    log(f'{name:<17} rewards {r}  mean {r_.mean():.3f}  std {r_.std():.3f}  ->  advantages {np.round(a, 3).tolist()}')
    OUT['cases'][name] = dict(r=r, mean=float(r_.mean()), std=float(r_.std()), adv=np.round(a, 4).tolist())
log('(with the sample std, dividing by G-1 = 3, the 2-right case gives +-0.866 instead of +-1: implementations differ)')

# ---------------------------------------------------------------- Part B
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).to(DEV)
SYSTEM = 'Please reason step by step, and put your final answer within \\boxed{}.'
QUESTIONS = [('Tom has 3 boxes with 12 apples in each box. He gives away 10 apples. How many apples does he have left?', 26),
             ('A train travels at 60 km per hour for 2.5 hours. How many kilometres does it travel?', 150),
             ('What is 17 times 23?', 391),
             ('A book costs 8 dollars and a pen costs 3 dollars. How much do 4 books and 5 pens cost in total?', 47)]
G = 4
END = tok.convert_tokens_to_ids('<|im_end|>')


def final_number(text):
    """The checker: take what is inside the last \\boxed{...}, else the last number in the text."""
    box = re.findall(r'\\boxed\{([^}]*)\}', text)
    src = box[-1] if box else text
    nums = re.findall(r'-?\d[\d,]*\.?\d*', src)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(',', '').rstrip('.'))
    except ValueError:
        return None


def prompt_ids(q):
    return tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}],
                                   add_generation_prompt=True, tokenize=True, return_dict=False)


def mean_logp(q, new):
    p = prompt_ids(q)
    ids = torch.tensor([p + new], device=DEV)
    lp = torch.log_softmax(model(ids).logits[0, len(p) - 1:-1].float(), -1)
    return lp.gather(1, torch.tensor(new, device=DEV)[:, None]).mean()     # average over the answer's tokens


torch.manual_seed(1)
groups = []
log(f'\n== B. real groups: {NAME}, G = {G} samples per question, temperature 0.8 ==')
for q, gold in QUESTIONS:
    p = prompt_ids(q)
    with torch.no_grad():
        out = model.generate(torch.tensor([p] * G, device=DEV), do_sample=True, temperature=0.8, top_p=1.0, top_k=0,
                             max_new_tokens=300, pad_token_id=tok.convert_tokens_to_ids('<|endoftext|>'), eos_token_id=END)
    samples = []
    for o in out:
        new = o[len(p):].tolist()
        new = new[:new.index(END) + 1] if END in new else new
        text = tok.decode(new, skip_special_tokens=True)
        pred = final_number(text)
        samples.append(dict(new=new, text=text, pred=pred, reward=float(pred is not None and abs(pred - gold) < 1e-6)))
    r = [s['reward'] for s in samples]
    a = advantages(r)
    log(f'\nQ: {q}   (correct answer {gold})')
    for i, (s, ai) in enumerate(zip(samples, a)):
        s['adv'] = float(ai)
        tail = s['text'].replace('\n', ' ')[-70:]
        log(f'  o{i + 1}: {len(s["new"]):>3} tokens  final answer {s["pred"]}  reward {s["reward"]:.0f}  advantage {ai:+.3f}   "...{tail}"')
    log(f'  rewards {r}  mean {np.mean(r):.2f}  std {np.std(r):.2f}')
    groups.append(dict(q=q, gold=gold, samples=samples))

# one policy-gradient step on all groups: loss = - mean_i A_i * (mean log-prob of o_i). On-policy, so PPO's ratio is 1
# and clipping does nothing on this first step; the KL term to the reference is 0 at the start, so it is left out here.
log('\n== C. one GRPO-style gradient step (learning rate 2e-6) ==')
before = [[float(mean_logp(g['q'], s['new'])) for s in g['samples']] for g in groups]
opt = torch.optim.AdamW(model.parameters(), lr=2e-6, weight_decay=0.0)
loss = 0
n = 0
for g in groups:
    for s in g['samples']:
        if s['adv'] != 0:
            loss = loss - s['adv'] * mean_logp(g['q'], s['new'])
            n += 1
if n:
    (loss / n).backward()
    opt.step()
with torch.no_grad():
    after = [[float(mean_logp(g['q'], s['new'])) for s in g['samples']] for g in groups]
rows = []
for g, b, a in zip(groups, before, after):
    for i, (s, x, y) in enumerate(zip(g['samples'], b, a)):
        rows.append(dict(q=g['q'][:30], i=i + 1, reward=s['reward'], adv=s['adv'], before=x, after=y))
        log(f'  {g["q"][:34]:<34} o{i + 1}  reward {s["reward"]:.0f}  A {s["adv"]:+.2f}   mean log-prob per token {x:.4f} -> {y:.4f}  ({y - x:+.4f})')
up = [r_['after'] - r_['before'] for r_ in rows if r_['adv'] > 0]
dn = [r_['after'] - r_['before'] for r_ in rows if r_['adv'] < 0]
zero = [r_['after'] - r_['before'] for r_ in rows if r_['adv'] == 0]
log(f'average change: positive-advantage answers {np.mean(up) if up else float("nan"):+.4f}, '
    f'negative-advantage answers {np.mean(dn) if dn else float("nan"):+.4f}, zero-advantage answers {np.mean(zero) if zero else float("nan"):+.4f}')
for g in groups:
    for s in g['samples']:
        s.pop('new')
OUT['groups'] = groups
OUT['step'] = rows
save('ch2_grpo', OUT)
