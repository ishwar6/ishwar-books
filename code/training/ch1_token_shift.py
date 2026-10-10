"""Chapter 1, experiment 3: how much does post-training change the next-token distribution?
In the style of Lin et al. (2023), "The Unlocking Spell on Base LLMs" (URIAL, arXiv 2312.01552), section 2:
1. The instruct model answers each prompt with greedy decoding: o = o_1 ... o_T.
2. At every position t we feed the SAME context (prompt + o_1..o_{t-1}) to the base model and to the instruct model.
3. We find the rank of o_t in the base model's sorted list ("base rank"):
   rank 1 = unshifted, rank 2-3 = marginal, rank > 3 = shifted.
4. We also compute KL(P_instruct || P_base) at every position.
Context format A: exactly the instruct model's tokens (chat template). Format B: a plain "Question/Answer" text for the base
model, as a check that the result is not only about special tokens."""
import math, collections, torch
from common import Log, save
from ch1_models import load, greedy, BASE, INSTRUCT, DEV

log = Log('ch1_token_shift')
PROMPTS = [
    # knowledge
    'What is the capital of Australia?', 'Who wrote the novel Pride and Prejudice?', 'What is photosynthesis?',
    'Why is the sky blue?', 'What causes the seasons on Earth?', 'How many bones are in the adult human body?',
    'What is the boiling point of water at sea level in Celsius?', 'Explain what DNA is in two sentences.',
    'What is the difference between weather and climate?', 'Who painted the Mona Lisa?',
    # how-to and advice
    'How do I make a cup of tea?', 'Give me three tips for sleeping better.', 'How can I improve my handwriting?',
    'What should I pack for a weekend camping trip?', 'How do I stay focused while studying?',
    'Suggest a name for a small bakery.', 'How do I politely decline a meeting invitation?',
    'What are some good habits for saving money?',
    # writing
    'Write a haiku about rain.', 'Write a short poem about the ocean.', 'Write a two-sentence story about a lost key.',
    'Write a thank-you note to a teacher.', 'Describe a sunset in one paragraph.', 'Write a slogan for a bicycle shop.',
    # maths and reasoning
    'What is 17 times 23?', 'A shop sells pencils at 3 for 45 cents. How much do 7 pencils cost?',
    'If a train travels 60 km in 45 minutes, what is its speed in km per hour?', 'Is 97 a prime number?',
    'What is the next number in the sequence 2, 4, 8, 16?', 'Convert 5 kilometres to miles.',
    # code
    'Write a Python function that reverses a string.', 'What does the SQL keyword JOIN do?',
    'Explain what a for loop is.', 'How do I read a file line by line in Python?',
    # conversation, identity, safety
    'Hello! How are you today?', 'Who are you?', 'Can you help me with my homework?', 'Tell me a joke.',
    'How do I pick a lock?', 'What is your opinion on pineapple on pizza?',
]
MAXN = 200
STOP = None

tok, inst = load(INSTRUCT)
_, base = load(BASE)
END = tok.convert_tokens_to_ids('<|im_end|>')


def chat(q):
    ids = tok.apply_chat_template([{'role': 'user', 'content': q}], add_generation_prompt=True, tokenize=True, return_dict=False)
    return ids['input_ids'] if isinstance(ids, dict) else ids


@torch.no_grad()
def dists(model, ids):
    """Log-probabilities of the next token at every position of ids: [T, V]."""
    return torch.log_softmax(model(torch.tensor([ids], device=DEV)).logits[0].float(), dim=-1)


positions = []      # one record per answer token
per_prompt = []
for pi, q in enumerate(PROMPTS):
    ctx = chat(q)
    ans, stopped = greedy(tok, inst, ctx, MAXN)
    full = ctx + ans
    li = dists(inst, full)[len(ctx) - 1: len(full) - 1]          # row t predicts ans[t]
    lb = dists(base, full)[len(ctx) - 1: len(full) - 1]
    # format B: the base model sees "Question: ...\nAnswer:" + the same answer tokens
    pb_ids = tok(f'Question: {q}\nAnswer:')['input_ids']
    ans_b = tok(' ' + tok.decode(ans[:-1] if stopped else ans))['input_ids']   # re-tokenised answer text, with a leading space
    lbB = dists(base, pb_ids + ans_b)[len(pb_ids) - 1: len(pb_ids) + len(ans_b) - 1]
    ranksB = [(lbB[t] > lbB[t, a]).sum().item() + 1 for t, a in enumerate(ans_b)]
    counts = collections.Counter()
    for t, a in enumerate(ans):
        pi_, pb_ = li[t].exp(), lb[t].exp()
        rank = (lb[t] > lb[t, a]).sum().item() + 1
        kl = (pi_ * (li[t] - lb[t])).sum().item()
        cls = 'unshifted' if rank == 1 else ('marginal' if rank <= 3 else 'shifted')
        counts[cls] += 1
        positions.append(dict(prompt=pi, t=t, tok=tok.decode([a]), rank=rank, cls=cls, kl=kl, before=tok.decode(ans[max(0, t - 8):t]),
                              p_inst=pi_[a].item(), p_base=pb_[a].item(), base_top=tok.decode([lb[t].argmax().item()]),
                              p_base_top=pb_.max().item()))
    n = len(ans)
    per_prompt.append(dict(prompt=q, answer=tok.decode(ans), n=n, stopped=stopped, **{k: counts[k] for k in ['unshifted', 'marginal', 'shifted']},
                           unshifted_B=sum(r == 1 for r in ranksB), top3_B=sum(r <= 3 for r in ranksB), n_B=len(ranksB)))
    log(f'[{pi + 1:>2}/{len(PROMPTS)}] {q[:48]!r:<52} {n:>3} tokens  unshifted {counts["unshifted"] / n:5.1%}  '
        f'marginal {counts["marginal"] / n:5.1%}  shifted {counts["shifted"] / n:5.1%}')

N = len(positions)
frac = {c: sum(p['cls'] == c for p in positions) / N for c in ['unshifted', 'marginal', 'shifted']}
NB = sum(r['n_B'] for r in per_prompt)
fracB = dict(unshifted=sum(r['unshifted_B'] for r in per_prompt) / NB, top3=sum(r['top3_B'] for r in per_prompt) / NB)
log(f'\n=== {len(PROMPTS)} prompts, {N} answer tokens (format A: identical chat-template context) ===')
log(f'unshifted (instruct token is base top-1): {frac["unshifted"]:.1%}')
log(f'marginal  (base rank 2 or 3):              {frac["marginal"]:.1%}')
log(f'shifted   (base rank > 3):                 {frac["shifted"]:.1%}')
log(f'within base top-3:                         {frac["unshifted"] + frac["marginal"]:.1%}')
log(f'format B (base sees "Question: ... Answer:"), {NB} tokens: unshifted {fracB["unshifted"]:.1%}, within top-3 {fracB["top3"]:.1%}')

kls = sorted(p['kl'] for p in positions)
mean_kl = sum(kls) / N
log(f'\nKL(P_instruct || P_base) per token: mean {mean_kl:.3f} nats, median {kls[N // 2]:.3f}, '
    f'90th percentile {kls[int(0.9 * N)]:.3f}, max {kls[-1]:.2f}')
for c in ['unshifted', 'marginal', 'shifted']:
    v = [p['kl'] for p in positions if p['cls'] == c]
    log(f'  mean KL at {c:<9} positions: {sum(v) / len(v):.3f}  ({len(v)} tokens)')
share = sum(p['kl'] for p in positions if p['cls'] == 'shifted') / sum(kls)
log(f'  shifted positions are {frac["shifted"]:.1%} of tokens but carry {share:.1%} of the total KL')

# by position in the answer
buckets = [(0, 5), (5, 20), (20, 50), (50, 100), (100, 200)]
by_pos = []
log('\nby position in the answer:')
for a, b in buckets:
    v = [p for p in positions if a <= p['t'] < b]
    if v:
        r = dict(range=f'{a + 1}-{b}', n=len(v), shifted=sum(p['cls'] == 'shifted' for p in v) / len(v),
                 unshifted=sum(p['cls'] == 'unshifted' for p in v) / len(v), kl=sum(p['kl'] for p in v) / len(v))
        by_pos.append(r)
        log(f'  tokens {r["range"]:>7}: n={r["n"]:>4}  unshifted {r["unshifted"]:.1%}  shifted {r["shifted"]:.1%}  mean KL {r["kl"]:.3f}')

# which tokens shift
sh = [p for p in positions if p['cls'] == 'shifted']
common = collections.Counter(p['tok'].strip() or repr(p['tok']) for p in sh).most_common(30)
log('\nmost common shifted tokens: ' + ', '.join(f'{t!r} x{c}' for t, c in common))
log('\nhighest-KL positions (instruct token | what the base model wanted | KL):')
for p in sorted(positions, key=lambda p: -p['kl'])[:25]:
    log(f'  {PROMPTS[p["prompt"]][:34]!r:<38} t={p["t"]:>3}  {p["tok"]!r:<14} base top-1 {p["base_top"]!r:<14} '
        f'P_inst {p["p_inst"]:.2f}  P_base {p["p_base"]:.3f}  KL {p["kl"]:.2f}')

# the end-of-turn token: does the base model know when to stop?
ends = [p for p in positions if p['tok'] == '<|im_end|>']
if ends:
    log(f'\nend-of-turn token <|im_end|> ({len(ends)} answers ended with it): mean P_instruct {sum(p["p_inst"] for p in ends) / len(ends):.3f}, '
        f'mean P_base {sum(p["p_base"] for p in ends) / len(ends):.4f}, '
        f'shifted in {sum(p["cls"] == "shifted" for p in ends)} of {len(ends)}; most common base top-1 there: '
        f'{collections.Counter(p["base_top"] for p in ends).most_common(3)}')

log('\nevery shifted position except the end-of-turn token (the 8 tokens before it | instruct token | base top-1):')
for p in sh:
    if p['tok'] != '<|im_end|>':
        log(f'  [{p["prompt"]:>2}] ...{p["before"][-40:]!r:<44} {p["tok"]!r:<12} base wanted {p["base_top"]!r}')

# a few unshifted content tokens, to show what does not move
content = [p for p in positions if p['cls'] == 'unshifted' and p['tok'].strip().istitle() and len(p['tok'].strip()) > 3]
log('\nexamples of unshifted content words: ' + ', '.join(sorted({p['tok'].strip() for p in content})[:40]))

# a token-by-token strip for one answer (for the figure)
ex = 0
strip = [dict(tok=p['tok'], cls=p['cls'], kl=p['kl'], rank=p['rank'], base_top=p['base_top'])
         for p in positions if p['prompt'] == ex]
save('ch1_token_shift', dict(n_prompts=len(PROMPTS), n_tokens=N, frac=frac, fracB=fracB, n_tokens_B=NB, mean_kl=mean_kl,
                             median_kl=kls[N // 2], p90_kl=kls[int(0.9 * N)], kl_share_shifted=share, by_pos=by_pos,
                             common_shifted=common, per_prompt=per_prompt, shifted=sh,
                             top_kl=sorted(positions, key=lambda p: -p['kl'])[:40], strip_prompt=PROMPTS[ex], strip=strip,
                             strip_all=[dict(prompt_text=PROMPTS[p['prompt']], tok=p['tok'], cls=p['cls'], kl=p['kl'], rank=p['rank'],
                                             base_top=p['base_top'], t=p['t'], p_inst=p['p_inst'], p_base=p['p_base']) for p in positions],
                             ends=dict(n=len(ends), p_inst=sum(p['p_inst'] for p in ends) / max(1, len(ends)),
                                       p_base=sum(p['p_base'] for p in ends) / max(1, len(ends)))))
