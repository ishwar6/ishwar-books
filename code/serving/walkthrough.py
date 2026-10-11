"""The small worked examples in the text of Part 6, computed rather than typed (CPU, standard library only).
Run: python code/serving/walkthrough.py  -> results/walkthrough.json, results/walkthrough_stdout.txt"""
import math
from common import Log

log = Log('walkthrough')
OUT = {}

# 1. TPOT from a request's timestamps
ttft, e2e, n = 0.42, 6.8, 128
tpot = (e2e - ttft) / (n - 1)
log(f'1. TPOT: E2E {e2e} s, TTFT {ttft} s, {n} tokens -> ({e2e} - {ttft}) / ({n} - 1) = {tpot * 1000:.1f} ms per token')
OUT['tpot'] = tpot

# 2. M/M/1 at 90% utilisation
mu = 10.0
for rho in (0.5, 0.9, 0.99):
    lam = rho * mu
    w = 1 / (mu - lam)
    log(f'2. M/M/1, mu = {mu:.0f}/s, rho = {rho}: W = 1 / ({mu:.0f} - {lam:.1f}) = {w * 1000:.0f} ms = {w * mu:.0f}x the 100 ms service time; '
        f'p99 = ln(100) x W = {math.log(100) * w * 1000:.0f} ms')
    OUT[f'mm1_{rho}'] = w

# 3. Quantizing one row of four weights to 4 bits, per row and in groups of two
w = [0.12, -0.03, 0.05, 0.90]
def q(row, bits=4):
    qmax = 2 ** (bits - 1) - 1
    s = max(abs(x) for x in row) / qmax
    codes = [max(-qmax - 1, min(qmax, round(x / s))) for x in row]
    return s, codes, [c * s for c in codes]
s, c, v = q(w)
log(f'3. 4-bit, one scale for the row: s = {max(map(abs, w))} / 7 = {s:.4f}; codes {c}; values {[round(x, 3) for x in v]}')
g1, g2 = q(w[:2]), q(w[2:])
log(f'   groups of two: s = {g1[0]:.4f} -> codes {g1[1]}, values {[round(x, 3) for x in g1[2]]}; '
    f's = {g2[0]:.4f} -> codes {g2[1]}, values {[round(x, 3) for x in g2[2]]}')
OUT['quant_row'] = {'scale': s, 'codes': c, 'values': v, 'groups': [g1, g2]}

# 4. LoRA adapter as KV-cache tokens
adapter = 80 * 2**20
log(f'4. a rank-16 Llama 3.1 8B adapter, 80 MiB = {adapter / 131072:.0f} tokens of BF16 KV cache (128 KiB each)')
log.save(OUT)
