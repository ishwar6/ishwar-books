"""Part 4 experiment, step 1 (heavy): sample answers from two small models on the same MATH-500 problems.

- deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B: the released R1 distill (SFT on R1's outputs, see paper section 2.4).
  It is NOT R1-Zero. It shows the behaviour (long thinking, "Wait"), not the RL training curve.
- Qwen/Qwen2.5-Math-1.5B: the base model that the distill started from, prompted with the R1-Zero template (Table 1).
  This is the kind of model R1-Zero's RL starts from: the "step 0" picture.

Sampling follows the paper's evaluation setup (page 12): temperature 0.6, top-p 0.95, k samples per question.
Our limits (a laptop, not a cluster): N_PROBLEMS problems, K samples each, at most MAX_NEW new tokens per answer
(the paper allows 32,768). An answer that hits the limit is marked truncated.
Writes results/r1_part4_gen_<tag>.jsonl (one line per answer) and results/r1_part4_gen_stdout.txt.
Usage: python r1_part4.py            (env: N_PROBLEMS=20 K=4 MAX_NEW=2048 BATCH=40)
"""
import json, os, random, time
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import common

N_PROBLEMS = int(os.environ.get('N_PROBLEMS', 20))
K = int(os.environ.get('K', 4))
MAX_NEW = int(os.environ.get('MAX_NEW', 2048))
BATCH = int(os.environ.get('BATCH', 40))
TAG = os.environ.get('TAG', '')                      # '' for the real run, 'test' for a smoke test
SEED = 0
log = common.Log('r1_part4_gen' + (f'_{TAG}' if TAG else ''))
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'

INSTRUCTION = 'Please reason step by step, and put your final answer within \\boxed{}.'
# Table 1 of the paper, the R1-Zero template. "prompt" is replaced by the question.
R1_ZERO_TEMPLATE = (
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
    'The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. '
    'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, '
    'respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. '
    'User: {prompt}. Assistant:')

MODELS = [
    ('distill', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'),
    ('base', 'Qwen/Qwen2.5-Math-1.5B'),
]


def pick_problems():
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    idx = sorted(random.Random(SEED).sample(range(len(ds)), N_PROBLEMS))
    return [dict(idx=i, problem=ds[i]['problem'], answer=ds[i]['answer'], level=ds[i]['level'],
                 subject=ds[i]['subject']) for i in idx]


def build_prompt(tok, tag, problem):
    user = problem + '\n' + INSTRUCTION                  # the same words for both models
    if tag == 'distill':                                  # the distill was trained with its chat template
        return tok.apply_chat_template([{'role': 'user', 'content': user}], tokenize=False,
                                       add_generation_prompt=True)
    return R1_ZERO_TEMPLATE.format(prompt=user)           # the base model sees the R1-Zero template


def run_model(tag, name, problems):
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = 'left'                             # left padding so every row ends at the same column
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(dev).eval()
    # one row per (problem, sample): N_PROBLEMS * K rows
    rows = [(p, s) for p in problems for s in range(K)]
    out_path = os.path.join(common.RESULTS, f'r1_part4_gen_{tag}' + (f'_{TAG}' if TAG else '') + '.jsonl')
    f = open(out_path, 'w')
    stop = ['</answer>'] if tag == 'base' else None       # R1-Zero's rollout ends at </answer>
    t_all = common.timer()
    for b in range(0, len(rows), BATCH):
        chunk = rows[b:b + BATCH]
        prompts = [build_prompt(tok, tag, p['problem']) for p, _ in chunk]
        enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(dev)
        torch.manual_seed(SEED + b)
        t = common.timer()
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, temperature=0.6, top_p=0.95, max_new_tokens=MAX_NEW,
                                 pad_token_id=tok.pad_token_id, stop_strings=stop, tokenizer=tok)
        new = gen[:, enc['input_ids'].shape[1]:]          # [batch, new_tokens]
        for (p, s), ids in zip(chunk, new):
            ids = ids.tolist()
            n = len(ids)
            while n and ids[n - 1] == tok.pad_token_id:   # drop the padding after an early stop
                n -= 1
            text = tok.decode(ids[:n], skip_special_tokens=True)
            ended = n < MAX_NEW or ids[n - 1] == tok.eos_token_id or (stop and text.rstrip().endswith('</answer>'))
            f.write(json.dumps(dict(model=tag, idx=p['idx'], sample=s, n_tokens=n, truncated=not ended,
                                    text=text)) + '\n')
        f.flush()
        log(f'{tag}: batch {b // BATCH + 1}/{(len(rows) + BATCH - 1) // BATCH}: {len(chunk)} answers, '
            f'longest {new.shape[1]} tokens, {t():.0f} s')
    log(f'{tag}: {len(rows)} answers in {t_all() / 60:.1f} min -> {os.path.basename(out_path)}')
    del model
    if dev == 'mps':
        torch.mps.empty_cache()


if __name__ == '__main__':
    problems = pick_problems()
    common.save('r1_part4_problems' + (f'_{TAG}' if TAG else ''), problems)
    log(f'MATH-500: {N_PROBLEMS} problems (seed {SEED}), K = {K} samples each, temperature 0.6, top-p 0.95, '
        f'max_new_tokens {MAX_NEW}, device {dev}')
    log('levels:', ' '.join(str(p['level']) for p in problems))
    with common.heavy('part4'):
        for tag, name in MODELS:
            run_model(tag, name, problems)
