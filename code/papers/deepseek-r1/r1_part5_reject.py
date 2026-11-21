"""Part 5, §2.3.3: rejection sampling, then the paper's three filters, on the released DeepSeek-R1-Distill-Qwen-1.5B.

1. Take 10 problems: 5 from GSM8K (test) and 5 from MATH-500 (one per difficulty level), picked with a fixed seed.
2. Sample 8 answers to each (temperature 0.6, top-p 0.95, at most 2048 new tokens), inside the heavy-job lock.
3. Keep only the correct ones. Method 1: our own \\boxed{} reader and answer compare. Method 2: math-verify.
   The funnel uses math-verify; we print every case where the two methods disagree.
4. Apply the three filters of §2.3.3 with our own code: mixed languages (the language reward of r1_part5_lang.py
   below 0.99), long paragraphs (a paragraph of the thinking with more than LONG_PARA words), code blocks (```).
5. Print the count after each step and write the kept answers as a tiny SFT file: results/r1_part5_sft.jsonl.
The thresholds are ours: the paper does not say where "long" starts or how much mixing is too much."""
import json, os, random, re, time
import common
from r1_part5_lang import lang_reward_scratch

MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
N_SAMPLES, MAX_NEW, TEMP, TOP_P = 8, 2048, 0.6, 0.95
LONG_PARA, MIN_LANG = 120, 0.99
SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'   # the model card's advice

log = common.Log('r1_part5_reject')


# ---------------------------------------------------------------- the problems
def pick_problems():
    from datasets import load_dataset
    rng = random.Random(5)
    gsm = load_dataset('openai/gsm8k', 'main', split='test')
    out = []
    for i in rng.sample(range(len(gsm)), 5):
        out.append(dict(id=f'gsm8k-{i}', question=gsm[i]['question'], answer=gsm[i]['answer'].split('####')[-1].strip()))
    m5 = load_dataset('HuggingFaceH4/MATH-500', split='test')
    for level in range(1, 6):
        idx = [i for i in range(len(m5)) if m5[i]['level'] == level]
        i = rng.choice(idx)
        out.append(dict(id=f'math500-L{level}-{i}', question=m5[i]['problem'], answer=m5[i]['answer']))
    return out


# ---------------------------------------------------------------- Method 1: our own answer check
def last_boxed(text):
    """The content of the last \\boxed{...}, with nested braces matched by hand."""
    k = text.rfind('\\boxed{')
    if k < 0:
        return None
    i, depth, start = k + len('\\boxed{'), 1, k + len('\\boxed{')
    while i < len(text) and depth:
        depth += {'{': 1, '}': -1}.get(text[i], 0)
        i += 1
    return text[start:i - 1] if depth == 0 else None


def norm(s):
    """Make two ways of writing one answer look the same: no spaces, no \\left/\\right, \\dfrac -> \\frac, ..."""
    s = s.strip().strip('$').rstrip('.')
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    for a, b in [('\\left', ''), ('\\right', ''), ('\\dfrac', '\\frac'), ('\\tfrac', '\\frac'), ('\\!', ''),
                 ('^\\circ', ''), ('^{\\circ}', ''), ('\\%', ''), ('%', ''), ('\\$', ''), (' ', ''), (',', '')]:
        s = s.replace(a, b)
    return s


def correct_scratch(text, gold):
    pred = last_boxed(text)
    if pred is None:
        return None, False
    a, b = norm(pred), norm(gold)
    if a == b:
        return pred, True
    try:
        return pred, abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return pred, False


# ---------------------------------------------------------------- Method 2: math-verify
def correct_mv(text, gold):
    from math_verify import parse, verify
    try:
        return bool(verify(parse('$' + gold + '$') or parse(gold), parse(text)))
    except Exception:
        return False


# ---------------------------------------------------------------- the three filters of §2.3.3
def split_think(text):
    """Generated text starts after '<think>\\n' (the template adds it). Returns (thinking, summary) or None."""
    if '</think>' not in text:
        return None
    t, s = text.split('</think>', 1)
    return t.strip(), s.strip()


def paragraph_words(thinking):
    return [len(p.split()) for p in re.split(r'\n\s*\n', thinking) if p.strip()]


def filters(thinking):
    lc, _, _ = lang_reward_scratch(thinking, 'en')
    longest = max(paragraph_words(thinking) or [0])
    return dict(lang=round(lc, 4), longest_para=longest, code='```' in thinking)


def main():
    problems = pick_problems()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    samples = []
    with common.heavy('part5'):
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
        log(f'model {MODEL} on {dev}, bfloat16; {N_SAMPLES} samples per problem, '
            f'temperature {TEMP}, top-p {TOP_P}, max {MAX_NEW} new tokens')
        t_all = common.timer()
        for pi, p in enumerate(problems):
            torch.manual_seed(pi)
            prompt = tok.apply_chat_template([{'role': 'user', 'content': p['question'] + SUFFIX}],
                                             tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to(dev)
            t = common.timer()
            with torch.no_grad():
                out = model.generate(ids, do_sample=True, temperature=TEMP, top_p=TOP_P, max_new_tokens=MAX_NEW,
                                     num_return_sequences=N_SAMPLES, pad_token_id=tok.eos_token_id)
            new = out[:, ids.shape[1]:]
            for j in range(N_SAMPLES):
                row = new[j].tolist()
                ended = tok.eos_token_id in row
                n = row.index(tok.eos_token_id) if ended else len(row)
                text = tok.decode(row[:n], skip_special_tokens=False)
                samples.append(dict(problem=p['id'], sample=j, question=p['question'], gold=p['answer'],
                                    text=text, n_tokens=n, ended=ended))
            log(f'  [{pi + 1:2d}/10] {p["id"]:22s} gold={p["answer"][:20]!r:24s} {time.strftime("%H:%M:%S")} '
                f'{t():.0f}s')
        log(f'generation took {t_all() / 60:.1f} minutes')

    # ---------------------------------------------------------------- score and filter (no model needed)
    disagree = 0
    for s in samples:
        parts = split_think(s['text']) if s['ended'] else None
        s['finished'] = parts is not None
        s['pred'], s['ok_scratch'] = correct_scratch(parts[1] if parts else s['text'], s['gold'])
        s['ok_mv'] = correct_mv(parts[1], s['gold']) if parts else False
        if parts:
            s.update(filters(parts[0]))
            s['thinking'], s['summary'] = parts
        if s['finished'] and s['ok_scratch'] != s['ok_mv']:
            disagree += 1
            log(f'  methods disagree on {s["problem"]} #{s["sample"]}: ours={s["ok_scratch"]} math-verify={s["ok_mv"]} '
                f'pred={s["pred"]!r} gold={s["gold"]!r}')
    fin = [s for s in samples if s['finished']]
    log('')
    log(f'answer check on {len(fin)} finished answers: our checker and math-verify agree on {len(fin) - disagree}')
    log('')
    log('per problem: correct answers out of 8 (math-verify), mean length in tokens')
    for p in problems:
        ss = [s for s in samples if s['problem'] == p['id']]
        log(f'  {p["id"]:22s} correct {sum(s["ok_mv"] for s in ss)}/8   finished {sum(s["finished"] for s in ss)}/8   '
            f'mean tokens {sum(s["n_tokens"] for s in ss) / 8:.0f}')
    flat = [p['id'] for p in problems
            if len({s['ok_mv'] for s in samples if s['problem'] == p['id']}) == 1]
    log(f'groups where all 8 rewards are equal (GRPO advantage 0 for every answer): {len(flat)} of {len(problems)}')

    steps = [('sampled', samples)]
    cur = samples
    for name, keep in [('finished (</think> and an end token)', lambda s: s['finished']),
                       ('correct (math-verify)', lambda s: s['ok_mv']),
                       (f'no mixed language (language reward >= {MIN_LANG})', lambda s: s['lang'] >= MIN_LANG),
                       (f'no long paragraph (<= {LONG_PARA} words)', lambda s: s['longest_para'] <= LONG_PARA),
                       ('no code block', lambda s: not s['code'])]:
        cur = [s for s in cur if keep(s)]
        steps.append((name, cur))
    log('')
    log('the funnel')
    for name, ss in steps:
        log(f'  {len(ss):3d}  {name}   ({len({s["problem"] for s in ss})} problems)')
    corr = [s for s in samples if s['ok_mv']]
    log('')
    log(f'among the {len(corr)} correct answers: language reward min {min(s["lang"] for s in corr):.3f}; '
        f'longest paragraph max {max(s["longest_para"] for s in corr)} words; code blocks {sum(s["code"] for s in corr)}')
    paras = sorted(w for s in corr for w in paragraph_words(s['thinking']))
    log(f'paragraphs in their thinking: {len(paras)}, median {paras[len(paras) // 2]} words, '
        f'over {LONG_PARA} words: {sum(w > LONG_PARA for w in paras)}')

    kept = steps[-1][1]
    path = os.path.join(common.RESULTS, 'r1_part5_sft.jsonl')
    with open(path, 'w') as f:
        for s in kept:
            f.write(json.dumps({'id': f'{s["problem"]}-{s["sample"]}', 'messages': [
                {'role': 'user', 'content': s['question'] + SUFFIX},
                {'role': 'assistant', 'content': '<think>\n' + s['thinking'] + '\n</think>\n\n' + s['summary']}]},
                ensure_ascii=False) + '\n')
    log('')
    log(f'wrote {len(kept)} records to results/r1_part5_sft.jsonl')
    short = min(kept, key=lambda s: s['n_tokens'])
    rec = json.loads([l for l in open(path) if f'"{short["problem"]}-{short["sample"]}"' in l][0])
    a = rec['messages'][1]['content']
    log('the shortest record, with the middle of the thinking cut for space:')
    log(json.dumps({'id': rec['id'], 'messages': [rec['messages'][0],
                    {'role': 'assistant', 'content': a[:260] + ' [...] ' + a[-420:]}]}, ensure_ascii=False, indent=1))

    json.dump(samples, open(os.path.join(common.RESULTS, 'r1_part5_samples.json'), 'w'), indent=1)
    common.save('r1_part5_reject', dict(
        funnel=[[n, len(ss)] for n, ss in steps], disagree=disagree, n_finished=len(fin),
        per_problem={p['id']: sum(s['ok_mv'] for s in samples if s['problem'] == p['id']) for p in problems},
        thresholds=dict(long_para=LONG_PARA, min_lang=MIN_LANG)))


if __name__ == '__main__':
    main()
