"""Worked numbers for Parts 3 and 4 of "How AI Got Here" (reinforcement learning, and how language models are trained today).
numpy only. Every number the two articles quote from a worked example is printed here.
Run: python rl_history.py > results/rl_history_stdout.txt"""
import numpy as np

np.set_printoptions(precision=3, suppress=True)
GAMMA = 0.9


def head(t):
    print('\n' + '=' * 70 + '\n' + t + '\n' + '=' * 70)


# ---------------------------------------------------------------------------------------------
# A corridor of five cells: 0 is a pit (reward -1, the episode ends), 4 is the goal (reward +1, the
# episode ends). The agent starts in cell 2 and moves left or right. Every other step gives 0.
# ---------------------------------------------------------------------------------------------
N, PIT, GOAL = 5, 0, 4
ACTIONS = ['left', 'right']


def step(s, a):
    s2 = s - 1 if a == 0 else s + 1
    r = -1.0 if s2 == PIT else (1.0 if s2 == GOAL else 0.0)
    return s2, r, s2 in (PIT, GOAL)


head('1. The corridor: Bellman expectation equation for a random policy (left or right with 1/2 each)')
V = np.zeros(N)
for sweep in range(200):
    newV = V.copy()
    for s in (1, 2, 3):
        tot = 0.0
        for a in (0, 1):
            s2, r, done = step(s, a)
            tot += 0.5 * (r + GAMMA * (0 if done else V[s2]))
        newV[s] = tot
    V = newV
print('V_random(1..3) =', np.round(V[1:4], 4))
s3 = 0.5 * (1 + GAMMA * 0) + 0.5 * (0 + GAMMA * V[2])
print(f'check at cell 3: 0.5*(1 + 0.9*0) + 0.5*(0 + 0.9*V(2)) = {s3:.4f}')

head('2. Value iteration (Bellman optimality), gamma = 0.9')
V = np.zeros(N)
for sweep in range(1, 6):
    newV = V.copy()
    for s in (1, 2, 3):
        best = -1e9
        for a in (0, 1):
            s2, r, done = step(s, a)
            best = max(best, r + GAMMA * (0 if done else V[s2]))
        newV[s] = best
    V = newV
    print(f'sweep {sweep}: V(1), V(2), V(3) = {V[1]:.4f}, {V[2]:.4f}, {V[3]:.4f}')
Vstar = V.copy()
Qstar = np.zeros((N, 2))
for s in (1, 2, 3):
    for a in (0, 1):
        s2, r, done = step(s, a)
        Qstar[s, a] = r + GAMMA * (0 if done else Vstar[s2])
print('Q*(s, left), Q*(s, right):')
for s in (1, 2, 3):
    print(f'  cell {s}: {Qstar[s, 0]:.4f}, {Qstar[s, 1]:.4f}')

head('3. TD(0): one update by hand, alpha = 0.5')
Vt = np.zeros(N)
s, s2, r = 3, 4, 1.0
target = r + GAMMA * 0
print(f'V(3) <- 0 + 0.5 * (1 + 0.9*0 - 0) = {0 + 0.5 * (target - 0):.4f}')
Vt[3] = 0.5
target = 0 + GAMMA * Vt[3]
print(f'next episode, step 2 -> 3, reward 0: V(2) <- 0 + 0.5 * (0 + 0.9*0.5 - 0) = {0.5 * (target - 0):.4f}')

head('4. Q-learning on the corridor, alpha = 0.5, epsilon = 0.2, 500 episodes from cell 2, seed 0')
rng = np.random.default_rng(0)
Q = np.zeros((N, 2))
first = []
for ep in range(500):
    s, done = 2, False
    while not done:
        a = rng.integers(2) if rng.random() < 0.2 else int(np.argmax(Q[s] + 1e-9 * rng.random(2)))
        s2, r, done = step(s, a)
        tgt = r + (0 if done else GAMMA * Q[s2].max())
        old = Q[s, a]
        Q[s, a] += 0.5 * (tgt - Q[s, a])
        if len(first) < 6 and old != Q[s, a]:
            first.append((ep, s, ACTIONS[a], r, round(tgt, 4), round(old, 4), round(Q[s, a], 4)))
        s = s2
print('first updates that changed the table (episode, cell, action, reward, target, old Q, new Q):')
for f in first:
    print('  ', f)
print('learned Q(s, left), Q(s, right) after 500 episodes:')
for s in (1, 2, 3):
    print(f'  cell {s}: {Q[s, 0]:.4f}, {Q[s, 1]:.4f}   greedy: {ACTIONS[int(np.argmax(Q[s]))]}')
print('largest gap to Q*:', round(float(np.abs(Q[1:4] - Qstar[1:4]).max()), 4))
print('hand example: Q(3, right) <- 0 + 0.5 * (1 + 0 - 0) =', 0.5)
print('hand example: Q(2, right) with Q(3,right)=0.5: 0 + 0.5 * (0 + 0.9 * 0.5 - 0) =', 0.5 * (0 + 0.9 * 0.5))

head('5. DQN target with a frozen target network (numbers by hand)')
q_target_next = np.array([0.6, 1.4, 0.9])
r, q_online = 0.5, 1.2
y = r + GAMMA * q_target_next.max()
print(f'y = r + gamma * max Q_target(s\', .) = 0.5 + 0.9 * {q_target_next.max()} = {y:.4f}')
print(f'loss (y - Q(s,a))^2 = ({y:.4f} - {q_online})^2 = {(y - q_online) ** 2:.4f}')

head('6. Two-armed bandit: greedy vs epsilon-greedy. Arm A pays 1 with p = 0.4, arm B with p = 0.6. 1,000 pulls, 2,000 runs')
rng = np.random.default_rng(1)
P = np.array([0.4, 0.6])
for eps in (0.0, 0.1):
    tot, best = 0.0, 0.0
    for run in range(2000):
        q, n = np.zeros(2), np.zeros(2)
        for t in range(1000):
            if t < 2:
                a = t
            elif rng.random() < eps:
                a = rng.integers(2)
            else:
                a = int(np.argmax(q))
            r = float(rng.random() < P[a])
            n[a] += 1
            q[a] += (r - q[a]) / n[a]
            tot += r
        best += (np.argmax(q) == 1)
    print(f'epsilon = {eps}: mean reward per pull {tot / 2000 / 1000:.4f}, runs whose final pick is B: {best / 2000:.3f}')
print('always B would give 0.6 per pull')

head('7. REINFORCE on the bandit: softmax policy over two preferences theta, learning rate 0.1')
theta = np.zeros(2)
pi = np.exp(theta) / np.exp(theta).sum()
grad = -pi.copy(); grad[1] += 1          # gradient of log pi(B) for a pull of arm B
theta2 = theta + 0.1 * 1.0 * grad        # reward 1
pi2 = np.exp(theta2) / np.exp(theta2).sum()
print(f'start pi = {pi}; pull B, reward 1; grad log pi(B) = {grad}; theta -> {theta2}; pi(B) -> {pi2[1]:.4f}')
theta3 = theta + 0.1 * (1.0 - 0.5) * grad
pi3 = np.exp(theta3) / np.exp(theta3).sum()
print(f'with baseline b = 0.5: theta -> {theta3}; pi(B) -> {pi3[1]:.4f}')
grad_a = -pi.copy(); grad_a[0] += 1
theta4 = theta + 0.1 * (0.0 - 0.5) * grad_a
pi4 = np.exp(theta4) / np.exp(theta4).sum()
print(f'with baseline b = 0.5, pull A, reward 0: theta -> {theta4}; pi(B) -> {pi4[1]:.4f}')
for base in (False, True):
    rng = np.random.default_rng(2)
    finals, varg = [], []
    for run in range(500):
        th, b = np.zeros(2), 0.0
        for t in range(500):
            p = np.exp(th) / np.exp(th).sum()
            a = int(rng.random() < p[1])
            r = float(rng.random() < P[a])
            g = -p.copy(); g[a] += 1
            adv = r - (b if base else 0.0)
            if t < 50:
                varg.append(adv * g[1])
            th += 0.1 * adv * g
            b += 0.05 * (r - b)
        finals.append(np.exp(th[1]) / np.exp(th).sum())
    print(f'baseline={base}: after 500 pulls mean pi(B) = {np.mean(finals):.3f} over 500 runs; '
          f'variance of the gradient term in the first 50 pulls = {np.var(varg):.4f}')

head('8. Actor-critic: the advantage from a one-step TD error')
r, Vs, Vs2 = 0.0, 0.5, 0.81
A = r + GAMMA * Vs2 - Vs
print(f'A = r + gamma V(s\') - V(s) = 0 + 0.9 * 0.81 - 0.5 = {A:.4f}')

head('9. PPO clipped objective, epsilon = 0.2')
eps = 0.2
print('ratio | A=+1: unclipped, clipped, min | A=-1: unclipped, clipped, min')
for rt in (0.5, 0.8, 1.0, 1.1, 1.2, 1.5, 2.0):
    row = []
    for A in (1.0, -1.0):
        u = rt * A
        c = np.clip(rt, 1 - eps, 1 + eps) * A
        row += [u, c, min(u, c)]
    print(f'{rt:4.1f}  | {row[0]:+.2f} {row[1]:+.2f} {row[2]:+.2f} | {row[3]:+.2f} {row[4]:+.2f} {row[5]:+.2f}')
print('ratio from probabilities: new 0.30 / old 0.20 =', round(0.30 / 0.20, 4))

head('10. AlphaZero: PUCT selection and the training loss on small numbers')
c = 1.5
moves = [('a', 0.6, 10, 0.50), ('b', 0.3, 2, 0.40), ('c', 0.1, 0, 0.0)]   # name, prior P, visits N, mean value Q
Ntot = sum(m[2] for m in moves)
for name, P_, n, q in moves:
    u = c * P_ * np.sqrt(Ntot) / (1 + n)
    print(f'move {name}: Q={q:.2f}, U = 1.5*{P_}*sqrt({Ntot})/(1+{n}) = {u:.4f}, Q+U = {q + u:.4f}')
z, v = 1.0, 0.6
pi_t = np.array([0.7, 0.2, 0.1]); p = np.array([0.5, 0.3, 0.2])
ce = -(pi_t * np.log(p)).sum()
print(f'value term (z - v)^2 = (1 - 0.6)^2 = {(z - v) ** 2:.4f}')
print(f'policy term -sum pi log p = {ce:.4f}   (pi = {pi_t}, p = {p})')
print(f'total without the weight penalty = {(z - v) ** 2 + ce:.4f}')

head('11. Bradley-Terry reward-model loss')
sig = lambda x: 1 / (1 + np.exp(-x))
for gap in (2.0, 0.5, 0.0, -2.0):
    print(f'r_w - r_l = {gap:+.1f}: P(winner preferred) = {sig(gap):.4f}, loss = {-np.log(sig(gap)):.4f}')

head('12. The RLHF objective with a KL penalty, beta = 0.1, per answer')
for name, rm, lp, lpref in [('answer 1', 2.0, -12.0, -11.0), ('answer 2', 2.6, -14.0, -22.0)]:
    kl = lp - lpref
    print(f'{name}: reward {rm}, log pi = {lp}, log pi_ref = {lpref}, log ratio = {kl:+.1f}, '
          f'penalised reward = {rm} - 0.1 * {kl:.1f} = {rm - 0.1 * kl:.4f}')

head('13. DPO loss, beta = 0.1 (sequence log-probabilities)')
beta = 0.1
lw, lwr, ll, llr = -10.0, -12.0, -15.0, -14.0
hw, hl = beta * (lw - lwr), beta * (ll - llr)
print(f'implicit reward of winner = 0.1 * (-10 - (-12)) = {hw:.4f}')
print(f'implicit reward of loser  = 0.1 * (-15 - (-14)) = {hl:.4f}')
print(f'margin = {hw - hl:.4f}, sigma = {sig(hw - hl):.4f}, loss = {-np.log(sig(hw - hl)):.4f}')
print(f'at the start (policy = reference): margin 0, loss = ln 2 = {np.log(2):.4f}')
print(f'gradient weight sigma(-margin) at the start = 0.5, now = {sig(-(hw - hl)):.4f}')

head('14. Best-of-n: chance that at least one of n answers is right (p = 0.3 each), and the KL of best-of-n')
for n in (1, 2, 4, 8, 16, 64):
    print(f'n = {n:2d}: 1 - 0.7^n = {1 - 0.7 ** n:.4f}; KL(best-of-n || base) = log n - (n-1)/n = {np.log(n) - (n - 1) / n:.4f} nats')

head('15. GRPO: group-relative advantages for 8 answers to one maths question (1 = correct)')
R = np.array([1, 0, 0, 1, 1, 0, 0, 0], dtype=float)
mu, sd = R.mean(), R.std()
print(f'rewards {R.astype(int)}; mean = {mu:.4f}; std (population) = {sd:.4f}')
print('advantages (r - mean) / std:', np.round((R - mu) / sd, 4))
print('Dr. GRPO style (r - mean), no division by std:', np.round(R - mu, 4))
R2 = np.array([1, 1, 1, 1, 1, 1, 1, 0], dtype=float)
print(f'an easy question, 7 of 8 right: advantages {np.round((R2 - R2.mean()) / R2.std(), 4)}')
R3 = np.ones(8)
print('all 8 right: std = 0, every advantage is 0, the question teaches nothing')

head('16. The discounted return and UCB (small extra examples)')
print('return of rewards 0, 0, 1 with gamma 0.9: 0 + 0.9*0 + 0.81*1 =', round(0 + 0.9 * 0 + 0.9 ** 2 * 1, 4))
print('return of rewards 1, 1, 1, 1, ... forever with gamma 0.9: 1 / (1 - 0.9) =', round(1 / (1 - 0.9), 4))
t = 100
for name, q, n in [('arm A', 0.55, 90), ('arm B', 0.40, 10)]:
    bonus = np.sqrt(np.log(t) / n)
    print(f'UCB at t = {t}, c = 1: {name}: Q = {q}, N = {n}, bonus sqrt(ln {t} / {n}) = {bonus:.4f}, score = {q + bonus:.4f}')

head('17. pass@k (the unbiased estimator of Chen et al. 2021): n = 10 samples, c = 3 correct')
from math import comb
for k in (1, 5):
    print(f'pass@{k} = 1 - C(7,{k}) / C(10,{k}) = 1 - {comb(7, k)} / {comb(10, k)} = {1 - comb(7, k) / comb(10, k):.4f}')
print('Arena-style Bradley-Terry: two models whose scores differ by 1 -> P(A preferred) =', round(float(sig(1.0)), 4))
