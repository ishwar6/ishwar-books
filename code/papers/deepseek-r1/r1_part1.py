"""Part 1 (the big idea): run the real released DeepSeek-R1-Distill-Qwen-1.5B and its base model Qwen2.5-Math-1.5B
on the same maths questions.

  1. The two models: same architecture, same size, different training.
  2. Temperature and top-p, by hand on four toy logits, then on the model's real logits.
     Method 1: our own functions. Method 2: the transformers logits warpers. Check: they agree.
  3. Greedy decoding: our own token-by-token loop (with a KV cache) against model.generate. Check: the same tokens.
  4. GSM8K question 0 with the distilled model and our own sampling loop (T = 0.6, top-p = 0.95): <think> ... </think>,
     then the answer; count the thinking tokens; check the answer by code (Method 1 our own checker, Method 2 math-verify).
  5. The same question with model.generate and the same settings.
  6. The same question, the same words, given to the base model.
  7. A MATH-500 question, 8 samples from each model: pass@1 and majority voting, computed by code.
Limits: 128 new tokens for the greedy check, 2,048 for steps 4 and 5, 1,024 for the base model, 6,144 for the distilled model in step 7, float32 on the Apple GPU (mps), seed 0."""
import collections, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import TemperatureLogitsWarper, TopPLogitsWarper
from datasets import load_dataset
from math_verify import parse, verify
import common

log = common.Log('r1_part1')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
DISTILL, BASE = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Qwen/Qwen2.5-Math-1.5B'
T, TOP_P = 0.6, 0.95                     # the paper's evaluation settings (§3, page 12)
INSTR = 'Please reason step by step, and put your final answer within \\boxed{}.'   # the official repo's advice for maths
R = {}


# ------------------------------------------------------------------ Method 1: sampling, written by hand
def softmax(z):
    """z: (V,) logits -> (V,) probabilities that add up to 1."""
    e = torch.exp(z - z.max())           # subtract the max so exp never overflows
    return e / e.sum()


def temperature_probs(logits, temperature):
    """Divide the logits by the temperature, then softmax. T < 1 makes the top token more likely."""
    return softmax(logits / temperature)                                  # (V,)


def top_p_filter(p, top_p):
    """Keep the smallest set of top tokens whose probabilities add up to at least top_p; renormalise. p: (V,)."""
    order = torch.argsort(p, descending=True)                             # (V,) token ids, most likely first
    sorted_p = p[order]
    before = torch.cumsum(sorted_p, 0) - sorted_p                         # mass of the tokens ranked above each one
    keep_sorted = before < top_p                                          # keep a token while the set is still short of top_p
    keep = torch.zeros_like(p, dtype=torch.bool)
    keep[order[keep_sorted]] = True
    q = torch.where(keep, p, torch.zeros_like(p))
    return q / q.sum(), int(keep.sum())


def pick(logits, gen=None, greedy=False):
    """One decoding step: logits (V,) -> one token id."""
    if greedy:
        return int(torch.argmax(logits))
    p = temperature_probs(logits.float().cpu(), T)
    q, _ = top_p_filter(p, TOP_P)
    return int(torch.multinomial(q, 1, generator=gen))


@torch.no_grad()
def our_generate(model, ids, max_new, eos_id, greedy=False, seed=0):
    """Method 1: our own token-by-token loop. The first call reads the whole prompt and fills the KV cache;
    every later call feeds one token and reuses the cache. ids: (1, n) prompt token ids."""
    gen = torch.Generator().manual_seed(seed)
    out = model(input_ids=ids, use_cache=True)                            # prefill: logits (1, n, V)
    past, logits = out.past_key_values, out.logits[0, -1]
    new = []
    for _ in range(max_new):
        tok = pick(logits, gen, greedy)
        new.append(tok)
        if tok == eos_id:
            break
        out = model(input_ids=torch.tensor([[tok]], device=ids.device), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[0, -1]               # (V,) scores for the next token
    return new


# ------------------------------------------------------------------ answer checking
def last_boxed(text):
    """The content of the last \\boxed{...} in the text, with nested braces, or None."""
    i = text.rfind('\\boxed{')
    if i < 0:
        return None
    depth, j = 0, i + len('\\boxed{') - 1
    for k in range(j, len(text)):
        depth += {'{': 1, '}': -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
    return None


def norm(a):
    """Method 1 checker: drop spaces, dollar signs, \\text{...} units and a trailing full stop."""
    a = re.sub(r'\\text\{[^}]*\}', '', a or '')
    a = a.replace('\\$', '').replace('$', '').replace(' ', '').replace(',', '').rstrip('.')
    return a


def check(pred, gold):
    """Return (ours, math_verify): is the boxed answer equal to the gold answer?"""
    ours = pred is not None and norm(pred) == norm(gold)
    mv = pred is not None and bool(verify(parse('$' + gold + '$'), parse('$\\boxed{' + pred + '}$')))
    return ours, mv


def split_think(tok, new_ids):
    """Split generated ids at </think>: (thinking ids, answer ids, finished?)."""
    end = tok.convert_tokens_to_ids('</think>')
    if end in new_ids:
        k = new_ids.index(end)
        return new_ids[:k], new_ids[k + 1:], True
    return new_ids, [], False


def load(name):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(DEV).eval()
    return m, AutoTokenizer.from_pretrained(name)


def chat_ids(tok, q):
    s = tok.apply_chat_template([{'role': 'user', 'content': q + '\n' + INSTR}], tokenize=False, add_generation_prompt=True)
    return s, tok(s, add_special_tokens=False, return_tensors='pt').input_ids.to(DEV)


def plain_ids(tok, q):
    s = q + '\n' + INSTR + '\n'
    return s, tok(s, return_tensors='pt').input_ids.to(DEV)


def sync():
    if DEV == 'mps':
        torch.mps.synchronize()


gsm = load_dataset('openai/gsm8k', 'main', split='test')[0]
Q1, G1 = gsm['question'], gsm['answer'].split('####')[-1].strip()
m500 = load_dataset('HuggingFaceH4/MATH-500', split='test')[10]
Q2, G2 = m500['problem'], m500['answer']

with common.heavy('part1'):
    torch.manual_seed(0)
    # ---------------------------------------------------------------- 1. the two models
    log(f'== 1. The two models (float32 on {DEV})')
    model, tok = load(DISTILL)
    base, btok = load(BASE)
    for name, m in ((DISTILL, model), (BASE, base)):
        c = m.config
        n = sum(p.numel() for p in m.parameters())
        log(f'  {name:42s} params {n:,}  layers {c.num_hidden_layers}  d_model {c.hidden_size}  heads {c.num_attention_heads}  vocab {c.vocab_size:,}')
    diffw = (model.model.layers[0].mlp.up_proj.weight - base.model.layers[0].mlp.up_proj.weight).abs().mean().item()
    log(f'  same shapes: {all(a.shape == b.shape for a, b in zip(model.parameters(), base.parameters()))}; '
        f'mean |difference| of one weight matrix (layer 0 up_proj): {diffw:.5f}')
    R['params'] = sum(p.numel() for p in model.parameters())

    # ---------------------------------------------------------------- 2. sampling by hand
    log('')
    log('== 2. Temperature and top-p on four toy logits')
    toy = torch.tensor([2.0, 1.0, 0.5, -1.0])
    names = ['"18"', '"9"', '"16"', '"2"']
    p1, p06 = temperature_probs(toy, 1.0), temperature_probs(toy, T)
    q, kept = top_p_filter(p06, TOP_P)
    lib = TopPLogitsWarper(TOP_P)(None, TemperatureLogitsWarper(T)(None, toy[None].clone()))[0]   # Method 2
    q_lib = torch.softmax(lib, -1)
    log('  token   logit   p (T=1)   p (T=0.6)   after top-p 0.95   library')
    for i, nm in enumerate(names):
        log(f'  {nm:5s} {toy[i]:7.2f} {p1[i]:9.3f} {p06[i]:11.3f} {q[i]:18.3f} {q_lib[i]:9.3f}')
    log(f'  top-p keeps {kept} of 4 tokens; max |ours - library| = {(q - q_lib).abs().max():.1e}')
    R['toy'] = dict(names=names, logits=toy.tolist(), p1=p1.tolist(), p06=p06.tolist(), top_p=q.tolist())

    s1, ids1 = chat_ids(tok, Q1)
    with torch.no_grad():
        real = model(input_ids=ids1).logits[0, -1].float().cpu()                              # (V,)
    pr = temperature_probs(real, T)
    qr, kept_r = top_p_filter(pr, TOP_P)
    qr_lib = torch.softmax(TopPLogitsWarper(TOP_P)(None, TemperatureLogitsWarper(T)(None, real[None].clone()))[0], -1)
    top = torch.argsort(pr, descending=True)[:5]
    log('')
    log(f'  real logits: the first token the distilled model writes after the GSM8K prompt (vocabulary {real.numel():,})')
    for t_ in top:
        log(f'    {tok.decode([int(t_)])!r:14s} p (T=0.6) = {pr[t_]:.4f}')
    log(f'  top-p 0.95 keeps {kept_r} tokens of {real.numel():,}; max |ours - library| = {(qr - qr_lib).abs().max():.1e}')

    # ---------------------------------------------------------------- 3. greedy: our loop against generate
    log('')
    log('== 3. Greedy decoding, 128 new tokens: our loop (KV cache) against model.generate')
    ours = our_generate(model, ids1, 128, tok.eos_token_id, greedy=True)
    with torch.no_grad():
        lib_ids = model.generate(ids1, attention_mask=torch.ones_like(ids1), max_new_tokens=128, do_sample=False, pad_token_id=tok.eos_token_id)[0, ids1.shape[1]:].tolist()
    agree = sum(a == b for a, b in zip(ours, lib_ids))
    log(f'  our loop {len(ours)} tokens, generate {len(lib_ids)} tokens, equal at {agree} of {min(len(ours), len(lib_ids))} positions: {ours == lib_ids}')
    log(f'  first 12 tokens: {tok.convert_ids_to_tokens(ours[:12])}')

    # ---------------------------------------------------------------- 4. the distilled model, our sampling loop
    log('')
    log('== 4. GSM8K test question 0, distilled model, our own sampling loop (T 0.6, top-p 0.95, seed 0, max 2,048 new tokens)')
    log(f'  prompt ({ids1.shape[1]} tokens): {s1!r}')
    sync(); t0 = time.time()
    new = our_generate(model, ids1, 2048, tok.eos_token_id, seed=0)
    sync(); dt = time.time() - t0
    think, ans, done = split_think(tok, new)
    text = tok.decode(new, skip_special_tokens=False)
    log('  ---- output ----')
    for line in text.split('\n'):
        log('  | ' + line)
    log('  ---- end ----')
    pred = last_boxed(tok.decode(ans))
    c_ours, c_mv = check(pred, G1)
    log(f'  new tokens {len(new)} in {dt:.1f} s ({len(new) / dt:.1f} tokens/s); thinking tokens {len(think)}; '
        f'answer tokens {len(ans)}; closed </think>: {done}')
    log(f'  boxed answer {pred!r}, gold {G1!r}; correct (our checker) {c_ours}; correct (math-verify) {c_mv}')
    R['q1_ours'] = dict(think=len(think), answer=len(ans), total=len(new), pred=pred, correct=c_ours)

    # ---------------------------------------------------------------- 5. the same with model.generate
    log('')
    log('== 5. The same question with model.generate (do_sample, T 0.6, top-p 0.95, seed 0, max 2,048 new tokens)')
    torch.manual_seed(0)
    with torch.no_grad():
        g = model.generate(ids1, attention_mask=torch.ones_like(ids1), max_new_tokens=2048, do_sample=True, temperature=T, top_p=TOP_P, top_k=0,
                           pad_token_id=tok.eos_token_id)[0, ids1.shape[1]:].tolist()
    think, ans, done = split_think(tok, g)
    pred = last_boxed(tok.decode(ans))
    c_ours, c_mv = check(pred, G1)
    log(f'  new tokens {len(g)}; thinking tokens {len(think)}; answer tokens {len(ans)}; closed </think>: {done}')
    log(f'  answer part: {tok.decode(ans, skip_special_tokens=True).strip()[:400]!r}')
    log(f'  boxed answer {pred!r}; correct (our checker) {c_ours}; correct (math-verify) {c_mv}')
    R['q1_lib'] = dict(think=len(think), answer=len(ans), total=len(g), pred=pred, correct=c_ours)

    # ---------------------------------------------------------------- 6. the base model, same words
    log('')
    log('== 6. The same question, the same words, base model Qwen2.5-Math-1.5B (no chat format; T 0.6, top-p 0.95, seed 0, max 1,024)')
    sb, idsb = plain_ids(btok, Q1)
    log(f'  prompt ({idsb.shape[1]} tokens): {sb!r}')
    newb = our_generate(base, idsb, 1024, btok.eos_token_id, seed=0)
    textb = btok.decode(newb, skip_special_tokens=False)
    log('  ---- output ----')
    for line in textb.split('\n')[:60]:
        log('  | ' + line)
    if len(textb.split('\n')) > 60:
        log(f'  | ... ({len(textb.split(chr(10))) - 60} more lines)')
    log('  ---- end ----')
    boxes = re.findall(r'\\boxed\{([^{}]*)\}', textb)
    log(f'  new tokens {len(newb)}; stopped by end-of-text: {newb[-1] == btok.eos_token_id}; contains <think>: {"<think>" in textb}; '
        f'boxed answers found: {boxes[:6]}')
    fb = boxes[0] if boxes else None
    c_ours, c_mv = check(fb, G1)
    log(f'  first boxed answer {fb!r}; correct (our checker) {c_ours}; correct (math-verify) {c_mv}')
    R['q1_base'] = dict(total=len(newb), stopped=newb[-1] == btok.eos_token_id, boxes=boxes, correct=c_ours)

    # ---------------------------------------------------------------- 7. pass@1 and majority voting
    log('')
    log('== 7. MATH-500 problem 10, 8 samples per model (T 0.6, top-p 0.95, seed 0): pass@1 and majority voting')
    log(f'  problem: {Q2!r}   gold answer: {G2}')
    brute = next(n for n in range(30, 10**6, 30) if set(str(n)) <= {'0', '2'})
    log(f'  brute-force check of the gold answer: smallest multiple of 30 written with digits 0 and 2 only = {brute}')
    K = 8
    for label, m, tk, mk, mx in (('distilled', model, tok, chat_ids, 6144), ('base', base, btok, plain_ids, 1024)):
        _, ids = mk(tk, Q2)
        torch.manual_seed(0)
        sync(); t0 = time.time()
        with torch.no_grad():
            outs = m.generate(ids.repeat(K, 1), attention_mask=torch.ones_like(ids.repeat(K, 1)), max_new_tokens=mx, do_sample=True, temperature=T, top_p=TOP_P, top_k=0,
                              pad_token_id=tk.eos_token_id)[:, ids.shape[1]:].tolist()
        sync(); dt = time.time() - t0
        rows, votes = [], []
        log(f'  -- {label} model, max {mx:,} new tokens, {dt:.0f} s for the batch of {K}')
        for i, o in enumerate(outs):
            if tk.eos_token_id in o:
                o = o[:o.index(tk.eos_token_id) + 1]
            th, an, done = split_think(tk, o)
            txt = tk.decode(an if done else o, skip_special_tokens=True)
            if label == 'base':
                bx = re.findall(r'\\boxed\{([^{}]*)\}', txt)
                pred = bx[0] if bx else None
            else:
                pred = last_boxed(txt)
            ok, ok_mv = check(pred, G2)
            votes.append(norm(pred) if pred else None)
            waits = len(re.findall(r'\bWait\b', tk.decode(th, skip_special_tokens=True)))
            rows.append(dict(tokens=len(o), think=len(th) if done else None, pred=pred, correct=ok, waits=waits))
            log(f'  sample {i}: {len(o):5d} tokens, thinking {str(len(th)) if done else "-":>5s}, "Wait" x{waits:<3d} '
                f'answer {pred!r:12s} correct {ok} (math-verify {ok_mv})')
            if label == 'distilled' and done and ok and 'shown' not in R:
                R['shown'] = i
                log('    ---- the end of its thinking, then its answer ----')
                for line in tk.decode(th[-120:] + [tk.convert_tokens_to_ids('</think>')] + an, skip_special_tokens=True).split('\n'):
                    log('    | ' + line)
                log('    ---- end ----')
        p1 = sum(r['correct'] for r in rows) / K
        cnt = collections.Counter(v for v in votes if v)
        maj = cnt.most_common(1)[0][0] if cnt else None
        log(f'  pass@1 = {sum(r["correct"] for r in rows)}/{K} = {p1:.3f}; votes {dict(cnt)}; majority answer {maj!r}; '
            f'majority correct: {maj is not None and maj == norm(G2)}')
        R[f'q2_{label}'] = dict(rows=rows, pass1=p1, votes=dict(cnt), majority=maj)

common.save('r1_part1', R)
log('')
log('saved results/r1_part1.json')
