"""Chapter 6 (first half): the arithmetic of golden-set size. No model calls.
1. The binomial standard error of a pass rate and the 95% interval half-width for several set sizes.
2. How many examples one measurement needs for a +/- 5 point interval.
3. How many examples per version two INDEPENDENT sets need to detect a 5-point change (alpha 0.05 two-sided, power 0.8).
4. The same question when both versions run on the SAME examples (paired; McNemar's test), for a few discordance rates.
5. A simulation check of item 3: two versions at 80% and 85%, many repeated experiments, how often the gap is detected.
Usage: python ch6a_stats.py   ->  results/ch6a_stats.json and results/ch6a_stats_stdout.txt"""
import math, random
from common import Log, save

log = Log('ch6a_stats')
Z95, Z80 = 1.959964, 0.841621          # two-sided 95% and one-sided 80% normal quantiles

log('1. Standard error of a pass rate: SE = sqrt(p (1 - p) / n); a 95% interval is about p +/- 1.96 SE')
sizes = [20, 50, 100, 200, 400, 1000]
table = []
log(f'{"n":>6} ' + ' '.join(f'{"p=" + str(p):>14}' for p in (0.5, 0.8, 0.95)))
for n in sizes:
    row = {'n': n}
    cells = []
    for p in (0.5, 0.8, 0.95):
        se = math.sqrt(p * (1 - p) / n)
        row[str(p)] = {'se': se, 'half': Z95 * se}
        cells.append(f'+/-{Z95 * se * 100:5.1f} pts')
    table.append(row)
    log(f'{n:>6} ' + ' '.join(f'{c:>14}' for c in cells))
log('')

log('2. Examples needed for a 95% interval of +/- 5 points: n = 1.96^2 p (1 - p) / 0.05^2')
one = {}
for p in (0.5, 0.8, 0.9):
    n = Z95 ** 2 * p * (1 - p) / 0.05 ** 2
    one[str(p)] = math.ceil(n)
    log(f'  p = {p}: n = 3.84 x {p * (1 - p):.2f} / 0.0025 = {n:.1f} -> {math.ceil(n)}')
log('')

log('3. Two independent sets, detect 80% -> 85% (alpha 0.05 two-sided, power 0.8):')
p1, p2 = 0.80, 0.85
n_ind = (Z95 + Z80) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p2 - p1) ** 2
log(f'  n per version = (1.96 + 0.84)^2 x (0.80 x 0.20 + 0.85 x 0.15) / 0.05^2 = {(Z95 + Z80) ** 2:.2f} x {p1 * (1 - p1) + p2 * (1 - p2):.4f} / 0.0025 = {n_ind:.0f}')
log('')

log('4. Same examples for both versions (paired). Only the discordant examples (pass on one, fail on the other) carry information.')
log('   n = (1.96 sqrt(q) + 0.84 sqrt(q - d^2))^2 / d^2, where q = share of discordant examples and d = 0.05')
paired = {}
for q in (0.06, 0.10, 0.20):
    d = 0.05
    n = (Z95 * math.sqrt(q) + Z80 * math.sqrt(q - d * d)) ** 2 / (d * d)
    paired[str(q)] = math.ceil(n)
    log(f'  q = {q:.2f}: n = {n:.0f}')
log('')

log('5. Simulation of item 3: 2,000 experiments, each with n examples per version, true rates 80% and 85%,')
log('   "detected" = the 95% interval of the difference excludes zero.')
rng = random.Random(6)
sim = {}
for n in (50, 100, 300, 900):
    hits = 0
    for _ in range(2000):
        a = sum(rng.random() < p1 for _ in range(n)) / n
        b = sum(rng.random() < p2 for _ in range(n)) / n
        se = math.sqrt(a * (1 - a) / n + b * (1 - b) / n) or 1e-9
        hits += (b - a) / se > Z95
    sim[n] = hits / 2000
    log(f'  n = {n:>4} per version: detected in {hits / 20:.1f}% of experiments')
save('ch6a_stats', {'table': table, 'one_set_5pts': one, 'n_independent_80_85': n_ind, 'n_paired': paired, 'simulation': sim})
