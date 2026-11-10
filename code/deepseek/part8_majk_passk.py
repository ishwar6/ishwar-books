"""Part 8 experiment 2: a toy model of DeepSeekMath's finding "RL enhances Maj@K but not Pass@K".

Each of 2,000 toy questions has a distribution over 5 candidate answers; answer 0 is the correct one.
The "instruct" model samples from that distribution. The "RL" model is the same model with every distribution
sharpened: p_i^alpha / sum_j p_j^alpha with alpha > 1. Sharpening moves probability to whatever answer was
already most likely. It cannot invent a correct answer the model never gave.
Pass@K: at least one of K samples is correct (exact: 1 - (1 - p)^K).
Maj@K: the most common answer among K samples is correct (Monte Carlo, ties broken at random).
This is a simulation, not a model run. Numpy only, CPU, fixed seed."""
import numpy as np
from common import Log, save

log = Log('part8_majk_passk')
rng = np.random.default_rng(0)
Q, A, ALPHA, TRIALS = 2000, 5, 2.0, 400
KS = [1, 4, 8, 16, 32, 64]

# Toy "instruct" model: per question, softmax of random logits, with a per-question bonus on the correct answer
logits = rng.normal(0, 1.0, (Q, A))
skill = rng.uniform(-1.0, 3.5, Q)          # how strongly the model prefers the right answer on this question
logits[:, 0] += skill
P = np.exp(logits); P /= P.sum(1, keepdims=True)


def sharpen(P, a):
    S = P ** a
    return S / S.sum(1, keepdims=True)


def pass_at_k(P, k):
    return float(np.mean(1 - (1 - P[:, 0]) ** k))


def maj_at_k(P, k):
    hits = 0.0
    cum = P.cumsum(1)
    for _ in range(TRIALS // max(1, k // 8)):
        u = rng.random((Q, k, 1))
        draws = (u > cum[:, None, :]).sum(2)                       # Q x k sampled answer ids
        counts = np.stack([(draws == a).sum(1) for a in range(A)], 1)
        top = counts.max(1, keepdims=True)
        winners = counts == top
        pick = winners[:, 0] / winners.sum(1)                       # chance answer 0 wins a random tie-break
        hits += pick.mean()
    return float(hits / (TRIALS // max(1, k // 8)))


R = sharpen(P, ALPHA)
res = {'K': KS, 'alpha': ALPHA, 'questions': Q, 'answers': A}
for name, M in [('instruct', P), ('rl', R)]:
    res[name] = {'pass': [pass_at_k(M, k) for k in KS], 'maj': [maj_at_k(M, k) for k in KS]}

log(f'Toy model: {Q} questions, {A} answers each, RL = sharpen every distribution with alpha = {ALPHA}')
log(f'Mean probability of the correct answer: instruct {P[:, 0].mean():.3f}, RL {R[:, 0].mean():.3f}')
mode_ok = (P.argmax(1) == 0)
log(f'Questions where the correct answer is already the most likely one: {mode_ok.mean():.1%}')
log('')
log(f'{"K":>3}  {"Pass@K instr":>12} {"Pass@K RL":>10}   {"Maj@K instr":>11} {"Maj@K RL":>9}')
for i, k in enumerate(KS):
    log(f'{k:3d}  {res["instruct"]["pass"][i]:12.1%} {res["rl"]["pass"][i]:10.1%}   '
        f'{res["instruct"]["maj"][i]:11.1%} {res["rl"]["maj"][i]:9.1%}')

# Why: split questions by whether the correct answer is the mode
log('')
log('Split by question type (probability of the correct answer, instruct -> RL):')
for label, m in [('correct answer is the most likely', mode_ok), ('correct answer is NOT the most likely', ~mode_ok)]:
    log(f'  {label:38}: {P[m, 0].mean():.3f} -> {R[m, 0].mean():.3f}  ({m.sum()} questions)')
res['split'] = {'mode_ok_frac': float(mode_ok.mean()),
                'mode_ok': [float(P[mode_ok, 0].mean()), float(R[mode_ok, 0].mean())],
                'mode_bad': [float(P[~mode_ok, 0].mean()), float(R[~mode_ok, 0].mean())]}
save('part8_majk_passk', res)
