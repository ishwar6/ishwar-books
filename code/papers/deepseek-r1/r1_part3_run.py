"""Part 3: where R1-Zero's RL starts. Put real questions into the exact Table 1 template, sample 8 answers each from two
base models (no fine-tuning, no chat template), and score every answer with the rewards of r1_part3_rewards.py.

Models: Qwen/Qwen2.5-Math-1.5B (the base of DeepSeek-R1-Distill-Qwen-1.5B, see §2.4) and Qwen/Qwen2.5-0.5B.
Questions: GSM8K test 0 and 2, MATH-500 test 16 and 48. Sampling: temperature 1.0 (the rollout temperature of the later
long version of the paper), top-p 1.0, at most 1,024 new tokens, generation stops at '</answer>'. Seed 0 per question.
Output: results/r1_part3_run_stdout.txt and results/part3_run.json (every sample and every score)."""
import time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import common
from common import Log, save
from r1_part3_rewards import (extract_boxed, same_answer, format_reward_regex, format_reward_parser, parse, verify)
import re

log = Log('r1_part3_run')

TEMPLATE = ('A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
            'The assistant first thinks about the reasoning process in the mind and then provides the user with the '
            'answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, '
            'respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. '
            'User: {prompt}. Assistant:')      # Table 1, word for word; "prompt" is replaced by the question

gsm = load_dataset('openai/gsm8k', 'main', split='test')
m500 = load_dataset('HuggingFaceH4/MATH-500', split='test')
QUESTIONS = [('GSM8K 0', gsm[0]['question'], gsm[0]['answer'].split('####')[-1].strip()),
             ('GSM8K 2', gsm[2]['question'], gsm[2]['answer'].split('####')[-1].strip().replace(',', '')),
             ('MATH-500 16', m500[16]['problem'], m500[16]['answer']),
             ('MATH-500 48', m500[48]['problem'], m500[48]['answer'])]
MODELS = ['Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-0.5B']
N, MAX_NEW = 8, 1024


def final_answer(output):
    """What the accuracy reward reads: the <answer> block if there is one (its last box if it has a box),
    otherwise the last \\boxed{} anywhere. None if neither exists."""
    m = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if m:
        return extract_boxed(m.group(1)) or m.group(1).strip()
    return extract_boxed(output)


def score(output, gold):
    fmt = format_reward_parser(output)[0]                      # strict format: Method 2
    ans = final_answer(output)
    acc = float(ans is not None and same_answer(ans, gold))    # accuracy: Method 1
    lib = float(ans is not None and bool(verify(parse('$' + gold + '$'), parse('$' + ans + '$'))))   # Method 2
    return dict(fmt_regex=format_reward_regex(output), fmt=fmt, think=float('<think>' in output and '</think>' in output),
                answer_tag=float('<answer>' in output and '</answer>' in output), boxed=float('\\boxed' in output),
                acc=acc, acc_lib=lib, reward=acc + fmt, answer=ans)


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
RUN = {}
with common.heavy('part3'):
    for mid in MODELS:
        tok = AutoTokenizer.from_pretrained(mid)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16).to(device).eval()
        log(f'== {mid} on {device}, bfloat16, {N} samples per question, T=1.0, max {MAX_NEW} new tokens')
        RUN[mid] = []
        for qname, q, gold in QUESTIONS:
            prompt = TEMPLATE.format(prompt=q.strip().rstrip('.'))
            ids = tok(prompt, return_tensors='pt').to(device)
            torch.manual_seed(0)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW,
                                     num_return_sequences=N, stop_strings=['</answer>'], tokenizer=tok,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            secs = time.time() - t0
            new = gen[:, ids['input_ids'].shape[1]:]
            rows = []
            for k in range(N):
                n_tok = int((new[k] != (tok.pad_token_id or tok.eos_token_id)).sum())
                text = tok.decode(new[k], skip_special_tokens=True)
                s = score(text, gold)
                s.update(text=text, tokens=n_tok, hit_limit=n_tok >= MAX_NEW)
                rows.append(s)
            RUN[mid].append(dict(q=qname, gold=gold, question=q, samples=rows, secs=round(secs, 1)))
            c = {k: int(sum(r[k] for r in rows)) for k in ('think', 'answer_tag', 'fmt', 'boxed', 'acc', 'acc_lib')}
            log(f'{qname:12} gold {gold:12} think tags {c["think"]}/{N}  answer tags {c["answer_tag"]}/{N}  '
                f'strict format {c["fmt"]}/{N}  correct {c["acc"]}/{N} (math-verify {c["acc_lib"]}/{N})  '
                f'hit {MAX_NEW} tokens {sum(r["hit_limit"] for r in rows)}/{N}  {secs:.0f} s')
            rw = torch.tensor([r['reward'] for r in rows])
            adv = (rw - rw.mean()) / (rw.std() + 1e-4) if rw.std() > 0 else torch.zeros_like(rw)   # Eq. 3 of GRPO
            log(f'   rewards (accuracy + format): {[r["reward"] for r in rows]}')
            log(f'   GRPO advantages (Eq. 3)    : {[round(a, 2) for a in adv.tolist()]}')
            RUN[mid][-1]['adv'] = [round(a, 3) for a in adv.tolist()]
        del model
        if device == 'mps':
            torch.mps.empty_cache()

log('')
log('== totals over 4 questions x 8 samples = 32 answers per model')
TOT = {}
for mid, qs in RUN.items():
    rows = [r for q in qs for r in q['samples']]
    t = {k: int(sum(r[k] for r in rows)) for k in ('think', 'answer_tag', 'fmt', 'boxed', 'acc', 'acc_lib')}
    t['both'] = int(sum(r['fmt'] * r['acc'] for r in rows))
    t['mean_reward'] = round(sum(r['reward'] for r in rows) / len(rows), 3)
    t['groups_with_signal'] = sum(len({r['reward'] for r in q['samples']}) > 1 for q in qs)
    TOT[mid] = t
    log(f'{mid:24} think {t["think"]:2}/32  answer tags {t["answer_tag"]:2}/32  strict format {t["fmt"]:2}/32  '
        f'correct {t["acc"]:2}/32  format and correct {t["both"]:2}/32  mean reward {t["mean_reward"]}  '
        f'groups with different rewards {t["groups_with_signal"]}/4')
save('part3_run', dict(runs=RUN, totals=TOT, template=TEMPLATE))
log('saved results/part3_run.json')
