"""Background page, experiment (b): chain-of-thought and majority voting on a small chat model.

Qwen2.5-0.5B-Instruct on the first N questions of the GSM8K test set:
  1. direct answer (greedy, "give only the number"),
  2. chain of thought (greedy, "think step by step"),
  3. 16 sampled chains of thought per question (temperature 0.7): one sample vs a majority vote over k samples.
Answer check, Method 1: our own regex for the last number; Method 2: math-verify. Vote, Method 1: our own count;
Method 2: collections.Counter. Output: results/r1_partb_cot_stdout.txt and results/r1_partb_cot.json."""
import json, re, time
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify
import common

log = common.Log('r1_partb_cot')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
N, K, TEMP, MAX_COT, MAX_DIRECT = 30, 16, 0.7, 512, 16
DIRECT = '{q}\nGive only the final answer as a number. Do not explain.'
PREFILL = 'The answer is '   # the reply starts here, so the model must give the number at once
COT = "{q}\nLet's think step by step. At the end, write \"The answer is N.\" where N is a number."


def last_number(text):
    """Method 1: our own answer reader. Take "The answer is X" if present, else the last number in the text."""
    m = re.findall(r'answer is\s*\$?(-?[\d,]*\.?\d+)', text)
    nums = m or re.findall(r'-?[\d,]*\.?\d+', text)
    if not nums:
        return None
    x = nums[-1].replace(',', '')
    try:
        v = float(x)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


def mv_correct(text, gold):
    """Method 2: math-verify reads the answer and compares it with the gold number."""
    return bool(verify(parse(gold), parse(text)))


def vote(answers):
    """Method 1: majority vote by hand. Count each answer; on a tie, the answer seen first wins."""
    counts = {}                                          # answer -> votes, in the order first seen
    for a in answers:
        if a is not None:
            counts[a] = counts.get(a, 0) + 1
    best = None
    for a, c in counts.items():                          # strict > keeps the earlier answer on a tie
        if best is None or c > counts[best]:
            best = a
    return best


def vote_lib(answers):
    """Method 2: the same vote with collections.Counter."""
    c = Counter(a for a in answers if a is not None)
    return c.most_common(1)[0][0] if c else None


def chat(tok, q, tmpl, prefill=''):
    return tok.apply_chat_template([{'role': 'user', 'content': tmpl.format(q=q)}], tokenize=False,
                                   add_generation_prompt=True) + prefill


def generate(model, tok, prompts, max_new, sample=False, n=1):
    enc = tok(prompts, return_tensors='pt', padding=True).to(DEV)
    kw = dict(do_sample=True, temperature=TEMP, top_p=1.0, top_k=0) if sample else dict(do_sample=False)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, num_return_sequences=n, pad_token_id=tok.pad_token_id, **kw)
    new = out[:, enc.input_ids.shape[1]:]
    lens = [(row != tok.pad_token_id).sum().item() for row in new]
    return tok.batch_decode(new, skip_special_tokens=True), lens


ds = load_dataset('openai/gsm8k', 'main', split='test').select(range(N))
qs = [d['question'] for d in ds]
gold = [d['answer'].split('####')[-1].strip().replace(',', '') for d in ds]

with common.heavy('partb'):
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(DEV).eval()
    t0 = time.time()
    direct, dlen = [], []
    for i in range(0, N, 10):
        o, l = generate(model, tok, [chat(tok, q, DIRECT, PREFILL) for q in qs[i:i + 10]], MAX_DIRECT)
        direct += o; dlen += l
    old = common.os.path.join(common.RESULTS, 'r1_partb_cot.json')
    reuse = common.os.environ.get('REUSE') and common.os.path.exists(old)
    if reuse:                                            # keep the long CoT and sample runs; redo only the direct answers
        J = json.load(open(old))
        cot, samples = J['replies']['cot'], J['replies']['samples']
        clen, slen = [J['clen']] * N, [[J['slen']] * K for _ in range(N)]
    cot, clen = (cot, clen) if reuse else ([], [])
    for i in (range(0) if reuse else range(0, N, 10)):
        o, l = generate(model, tok, [chat(tok, q, COT) for q in qs[i:i + 10]], MAX_COT)
        cot += o; clen += l
    samples, slen = (samples, slen) if reuse else ([], [])
    for i, q in (enumerate([]) if reuse else enumerate(qs)):
        torch.manual_seed(1000 + i)
        o, l = generate(model, tok, [chat(tok, q, COT)], MAX_COT, sample=True, n=K)
        samples.append(o); slen.append(l)
    secs = J['secs'] if reuse else time.time() - t0

# ---------------------------------------------------------------- scoring
d_ans = [last_number(t) for t in direct]
c_ans = [last_number(t) for t in cot]
s_ans = [[last_number(t) for t in row] for row in samples]
d_ok = [a == g for a, g in zip(d_ans, gold)]
c_ok = [a == g for a, g in zip(c_ans, gold)]
d_mv = [mv_correct(t, g) for t, g in zip(direct, gold)]
c_mv = [mv_correct(t, g) for t, g in zip(cot, gold)]
s_ok = [[a == g for a in row] for row, g in zip(s_ans, gold)]
s_mv = [[mv_correct(t, g) for t in row] for row, g in zip(samples, gold)]
one = sum(sum(r) for r in s_ok) / (N * K)
one_mv = sum(sum(r) for r in s_mv) / (N * K)
maj = {}
ties = 0
for k in [1, 2, 4, 8, 16]:
    v1 = [vote(row[:k]) for row in s_ans]
    v2 = [vote_lib(row[:k]) for row in s_ans]
    ties += sum(a != b for a, b in zip(v1, v2))
    maj[k] = sum(a == g for a, g in zip(v1, gold)) / N
any16 = sum(any(r) for r in s_ok) / N

log(f'== Chain of thought and majority vote: {MODEL}, first {N} GSM8K test questions, device {DEV}')
log(f'   direct: greedy, reply starts with {PREFILL!r}, max {MAX_DIRECT} new tokens; CoT: greedy, max {MAX_COT}; samples: {K} per question, '
    f'temperature {TEMP}, max {MAX_COT}')
log(f'   question 1: {qs[0][:90]}...  gold answer {gold[0]}')
log(f'   direct reply : {direct[0].strip()[:80]!r}')
log(f'   CoT reply    : {cot[0].strip()[:150]!r} ... {cot[0].strip()[-60:]!r}')
log('')
log('                              our regex   math-verify   mean new tokens')
log(f'  direct answer (greedy)       {sum(d_ok):2d}/{N}        {sum(d_mv):2d}/{N}         {sum(dlen) / N:6.1f}')
log(f'  step by step (greedy)        {sum(c_ok):2d}/{N}        {sum(c_mv):2d}/{N}         {sum(clen) / N:6.1f}')
log(f'  one sample (mean of {K})       {one * N:5.1f}/{N}     {one_mv * N:5.1f}/{N}       '
    f'{sum(sum(l) for l in slen) / (N * K):6.1f}')
for k in [1, 2, 4, 8, 16]:
    log(f'  majority vote of {k:2d} samples   {round(maj[k] * N):2d}/{N}')
log(f'  at least one of {K} right       {round(any16 * N):2d}/{N}')
log(f'  vote by hand vs Counter: {ties} differences over 5 values of k x {N} questions')
agree = sum(a == b for a, b in zip(d_ok + c_ok + sum(s_ok, []), d_mv + c_mv + sum(s_mv, [])))
log(f'  regex vs math-verify: same verdict on {agree} of {2 * N + N * K} replies')
hit = [i for i in range(N) if not c_ok[i] and maj[16] and vote(s_ans[i]) == gold[i]]
if hit:
    i = hit[0]
    log(f'  example, question {i + 1}: gold {gold[i]}; greedy CoT said {c_ans[i]}; 16 samples said {s_ans[i]}; vote {vote(s_ans[i])}')
log(f'  generation time {secs:.0f} s')

common.save('r1_partb_cot', dict(model=MODEL, N=N, K=K, temp=TEMP, direct=sum(d_ok), cot=sum(c_ok), one=one, maj=maj,
                                 any16=any16, dlen=sum(dlen) / N, clen=sum(clen) / N,
                                 slen=sum(sum(l) for l in slen) / (N * K), secs=secs,
                                 replies=dict(direct=direct, cot=cot, samples=samples, gold=gold)))
