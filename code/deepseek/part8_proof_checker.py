"""Part 8 experiment 3: what a proof checker gives a prover model, in miniature.

Lean 4 is not installed here. This toy checker plays its role for one theorem: for real a, b, (a + b)^2 >= 4ab.
A "proof" is a list of steps. Each step names a rule. The checker accepts or rejects every step, and the reward
is 1 only if every step is accepted and no step is left as `sorry` (a hole). This is the binary signal that
DeepSeek-Prover-V1.5 uses for RL ("RL from proof assistant feedback"). Sympy does the algebra. CPU only."""
import sympy as sp
from common import Log, save

log = Log('part8_proof_checker')
a, b = sp.symbols('a b', real=True)
GOAL = ((a + b) ** 2 - 4 * a * b, '>=', 0)


def check(step, prev):
    rule, expr = step
    if rule == 'sorry':
        return 'HOLE', 'left as sorry: accepted for now, proof not finished'
    if rule == 'rewrite':                       # the new expression must equal the previous one
        ok = sp.simplify(prev - expr) == 0
        return ('OK', 'equal after expanding') if ok else ('FAIL', f'not equal: difference is {sp.expand(prev - expr)}')
    if rule == 'sq_nonneg':                     # a square of a real expression is >= 0
        ok = isinstance(expr, sp.Pow) and expr.exp == 2 and sp.simplify(prev - expr) == 0
        return ('OK', 'a square is never negative') if ok else ('FAIL', 'not a square of the previous expression')
    return 'FAIL', 'unknown rule'


def run(name, steps):
    log(f'--- {name} ---')
    log(f'goal: {GOAL[0]} {GOAL[1]} {GOAL[2]}')
    prev, verdicts = GOAL[0], []
    for i, step in enumerate(steps, 1):
        v, why = check(step, prev)
        verdicts.append(v)
        log(f'step {i}: {step[0]:9} {str(step[1]):24} {v:4}  {why}')
        if v == 'FAIL':
            break
        if step[0] != 'sorry':
            prev = step[1]
    done = all(v == 'OK' for v in verdicts) and steps[-1][0] == 'sq_nonneg' and len(verdicts) == len(steps)
    reward = 1 if done else 0
    log(f'reward: {reward}\n')
    return {'name': name, 'verdicts': verdicts, 'reward': reward}


proofs = [
    ('a correct proof', [('rewrite', a**2 - 2*a*b + b**2), ('rewrite', (a - b)**2), ('sq_nonneg', (a - b)**2)]),
    ('a proof with one wrong step', [('rewrite', a**2 - 4*a*b + b**2), ('rewrite', (a - b)**2), ('sq_nonneg', (a - b)**2)]),
    ('a sketch with a hole (sorry)', [('sorry', a**2 - 2*a*b + b**2), ('rewrite', (a - b)**2), ('sq_nonneg', (a - b)**2)]),
]
out = [run(n, s) for n, s in proofs]
save('part8_proof_checker', out)
