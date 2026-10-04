"""Part 6: the worked numbers behind the equations of Part 6.

1. Sliding windows over a long text: how many 512-token windows, and how many attention entries, versus one long input.
2. Two blanks: the "independent" score p(a) p(b) that BERT's masked LM implies, versus the chain rule p(a) p(b | a).
3. Cosine similarity, written out for one real pair of [CLS] vectors.
4. Knowledge distillation: a teacher (bert-base-uncased) and a student (distilbert-base-uncased) on the same blank.
5. The pre-training budget of Appendix A.2 as arithmetic.
Runs on the CPU so the numbers are reproducible."""
import math
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from common import Log, save

torch.manual_seed(0)
log = Log('part6_math')
out = {}
tok = AutoTokenizer.from_pretrained('bert-base-uncased')

# ---------------------------------------------------------------- 1. sliding windows
log('1. Sliding windows over a long text')
long_text = ' '.join(['BERT reads the whole input at once, so every token can look at every other token.'] * 40)
N = len(tok(long_text, add_special_tokens=False).input_ids)          # content tokens, without [CLS] and [SEP]
L, overlap = 512, 128
w = L - 2                                                           # room for content in one window
step = w - overlap
n_formula = 1 if N <= w else math.ceil((N - w) / step) + 1
ids = tok(long_text, add_special_tokens=False).input_ids
starts = list(range(0, max(1, N - w + step), step))                 # slide by s until the end is covered
spans = [(a, min(N, a + w)) for a in starts]
n_tok = len(spans)
assert spans[-1][1] == N and n_tok == n_formula
lens = [b_ - a + 2 for a, b_ in spans]
log(f'   formula: number of windows = 1 + ceil((N - w) / s) = 1 + ceil(({N} - {w}) / {step}) = {n_formula}')
log(f'   by slicing the token list: {n_tok} windows, content tokens ' + ', '.join(f'{a}-{b_ - 1}' for a, b_ in spans) + f'; lengths with [CLS] and [SEP]: {lens}')
rows = []
for n in [N, 2000, 10000]:
    k = 1 if n <= w else math.ceil((n - w) / step) + 1
    one = (n + 2) ** 2
    win = k * L * L
    rows.append(dict(N=n, windows=k, one_long=one, windowed=win))
    log(f'   N = {n:>6}: windows = {k:>3}; attention entries per head per layer: one long input (N+2)^2 = {one:>11,}; windows padded to 512, k*512^2 = {win:>11,}; ratio {one / win:.2f}')
out['windows'] = dict(N=N, L=L, overlap=overlap, step=step, n_formula=n_formula, n_windows=n_tok, spans=spans, lengths=lens, table=rows)
log('')

# ---------------------------------------------------------------- 2. two blanks
log('2. Two blanks: independent guesses versus the chain rule')
mlm = AutoModelForMaskedLM.from_pretrained('bert-base-uncased').eval()


def probs(s):
    e = tok(s, return_tensors='pt')
    pos = (e.input_ids[0] == tok.mask_token_id).nonzero().flatten().tolist()
    with torch.no_grad():
        p = torch.softmax(mlm(**e).logits[0], -1)
    return [p[q] for q in pos]


p1, p2 = probs('i flew from [MASK] [MASK] to london last week.')
idx = lambda w_: tok.convert_tokens_to_ids(w_)
cands1 = [tok.decode([int(i)]) for i in p1.topk(10).indices]
pairs = []
for a in cands1:
    pa = float(p1[idx(a)])
    (pb_given_a,) = probs(f'i flew from {a} [MASK] to london last week.')
    b_top = tok.decode([int(pb_given_a.argmax())])
    for b in ['town', 'york', b_top]:
        pairs.append(dict(a=a, b=b, p_a=pa, p_b=float(p2[idx(b)]), p_b_given_a=float(pb_given_a[idx(b)])))
seen, uniq = set(), []
for r in pairs:
    if (r['a'], r['b']) not in seen:
        seen.add((r['a'], r['b'])); r['independent'] = r['p_a'] * r['p_b']; r['chain'] = r['p_a'] * r['p_b_given_a']; uniq.append(r)
best_ind = max(uniq, key=lambda r: r['independent'])
best_chain = max(uniq, key=lambda r: r['chain'])
for name in [('new', 'town'), ('new', 'york')]:
    r = next(x for x in uniq if (x['a'], x['b']) == name)
    log(f'   "{r["a"]} {r["b"]}":  p(a) = {r["p_a"]:.4f}, p(b) = {r["p_b"]:.4f}, p(b | a) = {r["p_b_given_a"]:.4f}')
    log(f'      independent p(a) p(b) = {r["independent"]:.6f};  chain rule p(a) p(b | a) = {r["chain"]:.6f}')
log(f'   best pair by the independent score (over {len(uniq)} candidate pairs): "{best_ind["a"]} {best_ind["b"]}" {best_ind["independent"]:.6f}')
log(f'   best pair by the chain rule:                                  "{best_chain["a"]} {best_chain["b"]}" {best_chain["chain"]:.6f}')
top_chain = sorted(uniq, key=lambda r: -r['chain'])[:5]
log('   top 5 by the chain rule: ' + ', '.join(f'{r["a"]} {r["b"]} {r["chain"]:.4f}' for r in top_chain))
out['two_blanks'] = dict(pairs=uniq, best_independent=best_ind, best_chain=best_chain, top_chain=top_chain)
log('')

# ---------------------------------------------------------------- 3. cosine similarity written out
log('3. Cosine similarity for one real pair of [CLS] vectors (bert-base-uncased)')
bert = AutoModel.from_pretrained('bert-base-uncased').eval()


def cls_vec(s):
    with torch.no_grad():
        return bert(**tok(s, return_tensors='pt')).last_hidden_state[0, 0]


a, b = cls_vec('A man is playing a guitar.'), cls_vec('A person is making music.')
dot, na, nb = float(a @ b), float(a.norm()), float(b.norm())
log(f'   a = C("A man is playing a guitar."), b = C("A person is making music."), each with {a.numel()} numbers')
log(f'   first 4 numbers: a = {[round(float(x), 3) for x in a[:4]]}, b = {[round(float(x), 3) for x in b[:4]]}')
log(f'   a . b = {dot:.3f};  |a| = {na:.3f};  |b| = {nb:.3f}')
log(f'   cos = a . b / (|a| |b|) = {dot:.3f} / ({na:.3f} x {nb:.3f}) = {dot / (na * nb):.3f}   (torch: {float(F.cosine_similarity(a, b, dim=0)):.3f})')
toy_a, toy_b = torch.tensor([3.0, 4.0]), torch.tensor([4.0, 3.0])
log(f'   toy check in 2-D: a = (3, 4), b = (4, 3): a . b = {float(toy_a @ toy_b):.0f}, |a| = |b| = 5, cos = 24 / 25 = {float(F.cosine_similarity(toy_a, toy_b, dim=0)):.2f}')
out['cosine'] = dict(dot=dot, norm_a=na, norm_b=nb, cos=dot / (na * nb), first4_a=[float(x) for x in a[:4]], first4_b=[float(x) for x in b[:4]])
log('')

# ---------------------------------------------------------------- 4. distillation
log('4. Knowledge distillation: teacher bert-base-uncased, student distilbert-base-uncased')
s_tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
student = AutoModelForMaskedLM.from_pretrained('distilbert-base-uncased').eval()
n_t = sum(p.numel() for p in mlm.bert.parameters())
n_s = sum(p.numel() for p in student.distilbert.parameters())
sent = 'i went to the [MASK] to deposit my paycheck.'


def dist(model, tk, T=1.0):
    e = tk(sent, return_tensors='pt')
    q = (e.input_ids[0] == tk.mask_token_id).nonzero().item()
    with torch.no_grad():
        return torch.softmax(model(**e).logits[0, q] / T, -1)


t, s = dist(mlm, tok), dist(student, s_tok)
assert tok.get_vocab() == s_tok.get_vocab()
ce_hard = float(-torch.log(s[idx('bank')]))
ce_soft = float(-(t * torch.log(s)).sum())
ent_t = float(-(t * torch.log(t)).sum())
log(f'   encoder parameters: teacher {n_t:,} (12 layers), student {n_s:,} (6 layers), ratio {n_s / n_t:.3f}')
log('   teacher top 5: ' + ', '.join(f'{tok.decode([int(i)])} {float(v):.3f}' for v, i in zip(*t.topk(5))))
log('   student top 5: ' + ', '.join(f'{tok.decode([int(i)])} {float(v):.3f}' for v, i in zip(*s.topk(5))))
log(f'   hard-label loss  -log s(bank)               = {ce_hard:.3f}')
log(f'   soft-target loss -sum_w t(w) log s(w)       = {ce_soft:.3f}   (teacher entropy, the smallest possible value: {ent_t:.3f})')
out['distill'] = dict(params_teacher=n_t, params_student=n_s, teacher_top5=[(tok.decode([int(i)]), float(v)) for v, i in zip(*t.topk(5))],
                      student_top5=[(tok.decode([int(i)]), float(v)) for v, i in zip(*s.topk(5))], ce_hard=ce_hard, ce_soft=ce_soft, teacher_entropy=ent_t)
log('')

# ---------------------------------------------------------------- 5. the pre-training budget (Appendix A.2 numbers)
log('5. The pre-training budget, from the numbers in Appendix A.2')
steps, seqs, length = 1_000_000, 256, 512
tokens_per_batch = seqs * length
total = steps * tokens_per_batch
log(f'   tokens per batch = 256 x 512 = {tokens_per_batch:,}  (the paper rounds to 128,000)')
log(f'   tokens seen      = 1,000,000 steps x {tokens_per_batch:,} = {total:,}  (an upper bound: 90% of steps use length 128)')
real = steps * seqs * (0.9 * 128 + 0.1 * 512)
log(f'   with 90% of steps at length 128 and 10% at 512: 1,000,000 x 256 x (0.9 x 128 + 0.1 x 512) = {real:,.0f}')
log(f'   predicted tokens (15%) of that: {0.15 * real:,.0f}')
log(f'   chip-days: BERT-base 16 chips x 4 days = {16 * 4}; BERT-large 64 chips x 4 days = {64 * 4}')
out['budget'] = dict(tokens_per_batch=tokens_per_batch, upper=total, mixed=real, predicted=0.15 * real, chip_days_base=64, chip_days_large=256)

save('part6_math', out)
