"""Part 6 (Table 5 on a laptop): the base model and its distilled version on the same MATH-500 problems.

Qwen/Qwen2.5-Math-1.5B (the base model of §2.4) against deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (the released
distilled model). 16 MATH-500 problems of levels 1 to 3 (seeded draw), 4 samples each, temperature 0.6 and top-p 0.95
as in §3, but at most 2048 new tokens (the paper allows 32,768). Each model gets its own chat template and the prompt
"Please reason step by step, and put your final answer within \\boxed{}." Answers are graded with math-verify
under the policy of r1_part6_regrade.py (the same as Part 4): a cut-off answer is wrong, and for the distilled model
only the text after </think> is graded. NOTE: results/part6_eval_stdout.txt was written by an older version that graded
the whole text; r1_part6_regrade.py regrades the saved answers, and the page uses its numbers.
Reports pass@1 (the mean correctness of the 4 samples), answer length in tokens, and how many answers hit the limit."""
import contextlib, json, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from common import Log, save, heavy, RESULTS
from r1_part6_regrade import grade_answer

MODELS = [('base', 'Qwen/Qwen2.5-Math-1.5B'), ('distilled', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B')]
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_PROBLEMS, K, MAX_NEW, BATCH_PROBLEMS = 16, 4, 2048, 4
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'
log = Log('part6_eval')


def pick_problems():
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    pool = [r for r in ds if r['level'] <= 3]
    return random.Random(6).sample(pool, N_PROBLEMS)


def prompt_text(tok, problem):
    """The base model's template already has the 'reason step by step' line as a system message; the R1 template
    has no system message, so the line goes into the user turn (the advice of the R1 model card)."""
    if 'Distill' in tok.name_or_path:
        msgs = [{'role': 'user', 'content': problem + SUFFIX}]
    else:
        msgs = [{'role': 'user', 'content': problem}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)




def run_model(tag, name, problems):
    tok = AutoTokenizer.from_pretrained(name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(DEV).eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.eos_token_id}   # every token that ends an answer
    if '<|im_end|>' in prompt_text(tok, 'x'):           # the Qwen template ends a turn with <|im_end|>
        eos.add(tok.convert_tokens_to_ids('<|im_end|>'))
    rows = []
    t0 = time.time()
    for b in range(0, len(problems), BATCH_PROBLEMS):
        chunk = problems[b:b + BATCH_PROBLEMS]
        texts = [prompt_text(tok, p['problem']) for p in chunk]
        enc = tok(texts, return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
        torch.manual_seed(1000 + b)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.6, top_p=0.95,
                             num_return_sequences=K, pad_token_id=tok.pad_token_id,
                             eos_token_id=sorted(eos))
        gen = out[:, enc['input_ids'].shape[1]:]                     # (BATCH_PROBLEMS * K, <= MAX_NEW)
        for i in range(gen.shape[0]):
            p = chunk[i // K]
            ids = gen[i].tolist()
            stops = [j for j, x in enumerate(ids) if x in eos]
            ended = bool(stops)
            n = stops[0] + 1 if ended else len(ids)                   # answer length, end token included
            text = tok.decode(ids[:n], skip_special_tokens=True)
            rows.append(dict(model=tag, uid=p['unique_id'], level=p['level'], gold=p['answer'], sample=i % K,
                             tokens=n, ended=ended, boxed='\\boxed' in text, correct=grade_answer(tag, p['answer'], text, ended),
                             text=text))
        done = rows[-len(chunk) * K:]
        log(f'  {tag:9s} problems {b + 1:2d}-{b + len(chunk):2d}: correct {sum(r["correct"] for r in done):2d}/{len(done)}, '
            f'mean tokens {sum(r["tokens"] for r in done) / len(done):6.0f}')
    log(f'  {tag:9s} generation time {time.time() - t0:.0f} s on {DEV}')
    del model
    if DEV == 'mps':
        torch.mps.empty_cache()
    return rows


def summary(rows, tag):
    r = [x for x in rows if x['model'] == tag]
    n = len(r)
    by_level = {}
    for lv in (1, 2, 3):
        s = [x for x in r if x['level'] == lv]
        if s:
            by_level[lv] = (sum(x['correct'] for x in s) / len(s), len(s) // K)
    solved_any = len({x['uid'] for x in r if x['correct']})
    ok = [x['tokens'] for x in r if x['correct']]
    return dict(model=tag, pass1=sum(x['correct'] for x in r) / n, mean_tokens=sum(x['tokens'] for x in r) / n,
                mean_tokens_correct=sum(ok) / max(1, len(ok)), hit_limit=sum(not x['ended'] for x in r),
                boxed=sum(x['boxed'] for x in r), samples=n, solved_any=solved_any, by_level=by_level)


def main(lock=True):
    """lock=False when r1_part6_run.py already holds the heavy-job lock."""
    log('$ python r1_part6_eval.py')
    problems = pick_problems()
    levels = [p['level'] for p in problems]
    log(f'== {N_PROBLEMS} MATH-500 problems (levels 1-3: {levels.count(1)} / {levels.count(2)} / {levels.count(3)}), '
        f'{K} samples each, T=0.6, top-p 0.95, max {MAX_NEW} new tokens, bf16 on {DEV}')
    rows = []
    with (heavy('part6') if lock else contextlib.nullcontext()):
        for tag, name in MODELS:
            log(f'== {name}')
            rows += run_model(tag, name, problems)
    with open(os.path.join(RESULTS, 'part6_eval_gen.jsonl'), 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    log('== summary (pass@1 = mean correctness over all samples)')
    S = {}
    for tag, _ in MODELS:
        s = summary(rows, tag)
        S[tag] = s
        lv = ', '.join(f'level {k}: {v[0]:.2f}' for k, v in s['by_level'].items())
        log(f'{tag:9s} pass@1 {s["pass1"]:.3f} ({sum(x["correct"] for x in rows if x["model"] == tag)}/{s["samples"]})'
            f' | problems solved at least once {s["solved_any"]}/{N_PROBLEMS} | {lv}')
        log(f'{"":9s} mean tokens {s["mean_tokens"]:.0f} (correct answers only: {s["mean_tokens_correct"]:.0f}) | '
            f'hit the {MAX_NEW}-token limit {s["hit_limit"]}/{s["samples"]} | has \\boxed{{}} {s["boxed"]}/{s["samples"]}')
    save('part6_eval', dict(summary=S, problems=[p['unique_id'] for p in problems], k=K, max_new=MAX_NEW))


if __name__ == '__main__':
    main()
