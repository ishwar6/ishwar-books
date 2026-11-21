"""Part 4: solve the Table 3 "aha moment" equation  sqrt(a - sqrt(a + x)) = x  (a > 1)  two ways.

Method 1, by hand in plain Python: square twice, see the quartic as a quadratic in a, read off four candidate roots,
and keep only the candidates that satisfy the ORIGINAL equation.
Method 2, with sympy (and numpy.roots): expand, factor, solve, and check by substitution.
The trap: the quartic x^4 - 2a x^2 - x + (a^2 - a) = 0 has no x^3 term, so its four roots add up to 0 (Vieta).
But squaring adds false roots. Only one root is real and satisfies the original equation."""
import math
import numpy as np
import sympy as sp
import common

log = common.Log('r1_part4_aha_math')


# ---------- Method 1: by hand ----------
def quartic_coeffs(a):
    """Coefficients of x^4 - 2a x^2 - x + (a^2 - a), highest power first: [1, 0, -2a, -1, a^2 - a]."""
    # square once:  a - sqrt(a + x) = x^2      ->  sqrt(a + x) = a - x^2
    # square again: a + x = (a - x^2)^2 = a^2 - 2a x^2 + x^4
    return [1, 0, -2 * a, -1, a * a - a]


def candidates(a):
    """The quartic is a quadratic in a:  a^2 - (2x^2 + 1) a + (x^4 - x) = 0.
    Its discriminant is (2x^2 + 1)^2 - 4(x^4 - x) = 4x^2 + 4x + 1 = (2x + 1)^2, a perfect square, so
    a = x^2 + x + 1  or  a = x^2 - x.  Each one is a quadratic in x with two roots: four candidates in all."""
    out = []
    d1 = 4 * a - 3                                   # x^2 + x + (1 - a) = 0
    if d1 >= 0:
        out += [(-1 + math.sqrt(d1)) / 2, (-1 - math.sqrt(d1)) / 2]
    d2 = 4 * a + 1                                   # x^2 - x - a = 0
    if d2 >= 0:
        out += [(1 + math.sqrt(d2)) / 2, (1 - math.sqrt(d2)) / 2]
    return out


def satisfies_original(a, x, tol=1e-9):
    """Plug x into sqrt(a - sqrt(a + x)) = x. Both square roots must exist and the result must equal x."""
    if a + x < 0:
        return False
    inner = a - math.sqrt(a + x)
    if inner < 0:
        return False
    return abs(math.sqrt(inner) - x) < tol


def method1(a):
    cands = candidates(a)
    good = [x for x in cands if satisfies_original(a, x)]
    return cands, good


# ---------- Method 2: sympy ----------
a, x = sp.symbols('a x', real=True)
quartic = sp.expand((a - x**2)**2 - (a + x))           # = 0 after squaring twice
factored = sp.factor(quartic)
answer = (sp.sqrt(4 * a - 3) - 1) / 2

if __name__ == '__main__':
    log('equation: sqrt(a - sqrt(a + x)) = x, with a > 1')
    log('after squaring twice (sympy):', sp.Eq(sp.collect(quartic, x), 0))
    log('factored (sympy):            ', sp.Eq(factored, 0))
    log('sum of the 4 quartic roots (Vieta, minus the x^3 coefficient): 0  <- the tempting wrong answer')
    for av in [3, 7, 13, 2.5]:
        cands, good = method1(av)
        roots = np.roots(quartic_coeffs(av))           # numpy: all four roots of the quartic
        log(f'a = {av}: Method 1 candidates {[round(c, 4) for c in cands]}, kept {[round(g, 4) for g in good]}')
        log(f'        numpy.roots {[round(float(r.real), 4) for r in sorted(roots, key=lambda r: -r.real)]}, '
            f'sum {float(roots.sum().real):+.4f}; formula (sqrt(4a-3)-1)/2 = {(math.sqrt(4 * av - 3) - 1) / 2:.4f}')
    # sympy: solve the original equation for one value of a, then check the general answer by substitution
    sol3 = sp.solve(sp.Eq(sp.sqrt(3 - sp.sqrt(3 + x)), x), x)
    log('sympy solve for a = 3:', sol3)
    # write a through s = sqrt(4a - 3) > 1, i.e. s = 1 + u with u > 0, so a = (s^2 + 3)/4 and X = (s - 1)/2
    u = sp.Symbol('u', positive=True)
    s_ = 1 + u
    a_s, X = (s_**2 + 3) / 4, (s_ - 1) / 2
    inner = sp.sqrt(sp.factor(sp.expand(a_s + X)))           # sqrt((u + 2)^2 / 4) = (u + 2)/2, since u > 0
    outer = sp.sqrt(sp.factor(sp.expand(a_s - inner)))       # sqrt(u^2 / 4) = u/2
    log('sympy: a + X =', sp.factor(sp.expand(a_s + X)), ' so sqrt(a + X) =', inner,
        '; a - sqrt(a + X) =', sp.factor(sp.expand(a_s - inner)), ' so the left side =', outer, ' and X =', X)
    log('sympy check: left side - X =', sp.simplify(outer - X), ' for every a > 1')
    log('answer: the only real solution is x = (sqrt(4a - 3) - 1)/2, so the sum is (sqrt(4a - 3) - 1)/2')
