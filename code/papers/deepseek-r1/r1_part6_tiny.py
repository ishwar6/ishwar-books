"""Part 6 (optional): a tiny distillation run on a laptop, the §2.3.3 + §2.4 recipe at toy scale.

Teacher: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (our stand-in for R1). Student: Qwen/Qwen2.5-0.5B-Instruct.
1. The teacher writes one trace for each of 128 GSM8K training problems (T=0.6, top-p 0.95, max 1024 new tokens).
   We keep only the traces with the right final answer: rejection sampling, as in §2.3.3.
2. SFT of the whole student on the kept traces, with the masked loss of r1_part6.py (prompt tokens get -100):
   AdamW, learning rate 1e-5 with cosine decay to one tenth (the shape of the later version's B.4.3), batch 1,
   3 epochs. Full fine-tuning in fp32 on mps.
3. Before and after on 30 held-out GSM8K test problems, 2 samples each (T=0.6, top-p 0.95, max 1024 new tokens):
   accuracy, answer length, and how many answers use the teacher's habits ('</think>', 'Wait')."""
import contextlib, math, random, re, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from math_verify import parse, verify
from common import Log, save, heavy
from r1_part6 import sft_loss_from_scratch, SUFFIX

TEACHER, STUDENT = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-0.5B-Instruct'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_TRAIN, N_TEST, K_TEST, MAX_NEW, EPOCHS, LR = 128, 30, 2, 1024, 3, 1e-5
log = Log('part6_tiny')


def gold_of(row):
    return row['answer'].split('####')[-1].strip().replace(',', '')


def correct(gold, text):
    try:
        return int(verify(parse('$' + gold + '$'), parse(text)))
    except Exception:
        return 0


def generate(model, tok, prompts, k, seed, batch=16):
    """Sample k answers for each prompt string. Returns a list of (prompt index, text, n_tokens, ended)."""
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'), tok.convert_tokens_to_ids('<｜end▁of▁sentence｜>')}
    eos = sorted(e for e in eos if isinstance(e, int) and e >= 0)
    out_rows = []
    per = max(1, batch // k)
    for b in range(0, len(prompts), per):
        enc = tok(prompts[b:b + per], return_tensors='pt', padding=True, add_special_tokens=False).to(DEV)
        torch.manual_seed(seed + b)
        out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.6, top_p=0.95,
                             num_return_sequences=k, pad_token_id=tok.pad_token_id, eos_token_id=eos)
        gen = out[:, enc['input_ids'].shape[1]:]
        for i in range(gen.shape[0]):
            ids = gen[i].tolist()
            stops = [j for j, x in enumerate(ids) if x in eos]
            n = stops[0] + 1 if stops else len(ids)
            out_rows.append((b + i // k, tok.decode(ids[:n], skip_special_tokens=True), n, bool(stops)))
    return out_rows


def student_prompt(tok, question):
    return tok.apply_chat_template([{'role': 'user', 'content': question + SUFFIX}], add_generation_prompt=True,
                                   tokenize=False)


def evaluate(model, tok, test, tag):
    prompts = [student_prompt(tok, r['question']) for r in test]
    rows = generate(model, tok, prompts, K_TEST, seed=7)
    acc = sum(correct(gold_of(test[i]), t) for i, t, _, _ in rows) / len(rows)
    toks = sum(n for _, _, n, _ in rows) / len(rows)
    think = sum('</think>' in t for _, t, _, _ in rows)
    wait = sum(bool(re.search(r'\bWait\b', t)) for _, t, _, _ in rows)
    limit = sum(not e for _, _, _, e in rows)
    log(f'{tag:7s} accuracy {acc:.3f} ({round(acc * len(rows))}/{len(rows)}) | mean tokens {toks:.0f} | '
        f'with </think> {think}/{len(rows)} | with "Wait" {wait}/{len(rows)} | hit the limit {limit}/{len(rows)}')
    return dict(acc=acc, tokens=toks, think=think, wait=wait, limit=limit, n=len(rows), sample=rows[0][1][:600])


def main(lock=True):
    """lock=False when r1_part6_run.py already holds the heavy-job lock."""
    log('$ python r1_part6_tiny.py')
    train_rows = load_dataset('openai/gsm8k', 'main', split='train')
    test_rows = load_dataset('openai/gsm8k', 'main', split='test')
    train = [train_rows[i] for i in random.Random(0).sample(range(len(train_rows)), N_TRAIN)]
    test = [test_rows[i] for i in random.Random(1).sample(range(1, len(test_rows)), N_TEST)]   # #0 is used elsewhere
    R = {}
    with (heavy('part6') if lock else contextlib.nullcontext()):
        t0 = time.time()
        # ---------------------------------------------------------------- 1. the teacher writes, we keep the right ones
        log(f'== 1. teacher {TEACHER} writes one trace per problem ({N_TRAIN} GSM8K train problems, max {MAX_NEW} tokens)')
        ttok = AutoTokenizer.from_pretrained(TEACHER, padding_side='left')
        teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to(DEV).eval()
        # 0. one position of our trace: the teacher's full next-token distribution (word-level target) against
        #    the one token the sampled text keeps (sequence-level target). Used by the figure of the two kinds.
        import json, os
        from common import RESULTS
        L = json.load(open(os.path.join(RESULTS, 'part6_loss.json')))
        from r1_part6 import build_example
        ids, labels, P = build_example(ttok, L['trace']['question'], L['trace']['text'])
        resp = ids[P:]
        words = [ttok.decode([i]) for i in resp]
        pos = next((k for k, w in enumerate(words) if w.strip() == 'Wait' and k > 0), 1)
        with torch.no_grad():
            logits = teacher(torch.tensor([ids[:P + pos]], device=DEV)).logits[0, -1].float()
        probs = logits.softmax(-1)
        top = probs.topk(5)
        R['teacher_top'] = dict(context=ttok.decode(resp[:pos]), picked=words[pos], picked_p=probs[resp[pos]].item(),
                                top=[(ttok.decode([i]), p) for i, p in zip(top.indices.tolist(), top.values.tolist())])
        log(f'== 0. teacher distribution at response position {pos} (after {R["teacher_top"]["context"][-40:]!r}):')
        log('   ' + ', '.join(f'{t!r} {p:.3f}' for t, p in R['teacher_top']['top']) +
            f'; the trace has {words[pos]!r} (p = {R["teacher_top"]["picked_p"]:.3f})')
        tprompts = [ttok.apply_chat_template([{'role': 'user', 'content': r['question'] + SUFFIX}],
                                             add_generation_prompt=True, tokenize=False) for r in train]
        traces = generate(teacher, ttok, tprompts, 1, seed=3)
        kept = [(train[i], t) for i, t, n, ended in traces if ended and correct(gold_of(train[i]), t)]
        log(f'traces written {len(traces)}, finished and correct (kept) {len(kept)}, '
            f'mean length of kept {sum(len(ttok(t)["input_ids"]) for _, t in kept) / len(kept):.0f} tokens, '
            f'{time.time() - t0:.0f} s')
        del teacher
        torch.mps.empty_cache() if DEV == 'mps' else None

        # ---------------------------------------------------------------- 2. SFT of the student
        stok = AutoTokenizer.from_pretrained(STUDENT, padding_side='left')
        student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32).to(DEV)
        student.eval()
        log(f'== 3a. student {STUDENT} before distillation ({N_TEST} held-out GSM8K test problems x {K_TEST} samples)')
        R['before'] = evaluate(student, stok, test, 'before')

        log(f'== 2. SFT on the kept traces: {EPOCHS} epochs, batch 1, AdamW, lr {LR} cosine to {LR / 10}')
        examples = []
        for row, trace in kept:
            p = stok(student_prompt(stok, row['question']), add_special_tokens=False)['input_ids']
            r = stok('<think>\n' + trace + '<|im_end|>', add_special_tokens=False)['input_ids']
            examples.append((p + r, [-100] * len(p) + r))
        steps = EPOCHS * len(examples)
        opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
        student.train()
        step, losses, t1 = 0, [], time.time()
        for ep in range(EPOCHS):
            order = list(range(len(examples)))
            random.Random(100 + ep).shuffle(order)
            for j in order:
                ids, lab = examples[j]
                ids_t = torch.tensor([ids], device=DEV)
                lab_t = torch.tensor(lab, device=DEV)
                logits = student(ids_t).logits[0].float()           # (T, V)
                loss, _ = sft_loss_from_scratch(logits, lab_t)      # our masked loss from Method 1
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                losses.append(loss.item()); step += 1
                if step % 25 == 0 or step == 1 or step == steps:
                    w = losses[-25:]
                    log(f'  step {step:4d}/{steps}  loss {sum(w) / len(w):.4f}  (mean of last {len(w)})  '
                        f'{time.time() - t1:.0f} s')
        R['losses'] = losses
        student.eval()
        log(f'== 3b. student after distillation')
        R['after'] = evaluate(student, stok, test, 'after')
        log(f'total time {time.time() - t0:.0f} s on {DEV}')
        log('a sample answer after distillation starts:', repr(R['after']['sample'][:300]))
    R['kept'], R['written'] = len(kept), len(traces)
    save('part6_tiny', R)


if __name__ == '__main__':
    main()
