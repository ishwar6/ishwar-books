"""Chapter 1, the arithmetic of agents: (1) if every step succeeds with probability p, a task of n steps succeeds with p^n;
(2) what an 8-step agent costs in tokens against a single call, at an EXAMPLE price (not a quote from any provider).
Writes results/ch1_math.json and results/ch1_math_stdout.txt."""
from common import Log, save

log = Log('ch1_math')
PS = [0.90, 0.95, 0.99, 0.999]
NS = [1, 5, 10, 20, 50]

log('== Reliability: P(all n steps succeed) = p^n ==')
log(f'{"p":>6} | ' + ' | '.join(f'{n:>3} steps' for n in NS))
rel = {}
for p in PS:
    rel[str(p)] = {str(n): p ** n for n in NS}
    log(f'{p:>6} | ' + ' | '.join(f'{100 * p ** n:7.1f}%' for n in NS))
log('')
log('Steps you can afford before success drops below 90% and 50%:')
limits = {}
for p in PS:
    n90 = next(n for n in range(1, 100000) if p ** n < 0.90) - 1
    n50 = next(n for n in range(1, 100000) if p ** n < 0.50) - 1
    limits[str(p)] = {'n90': n90, 'n50': n50}
    log(f'  p = {p}: {n90:>4} steps keep >= 90%, {n50:>4} steps keep >= 50%')
log('')

# ---- token cost. The price is an EXAMPLE used only to make the arithmetic concrete; check your provider's price list.
PRICE_IN, PRICE_OUT = 3.00, 15.00          # example: dollars per million input / output tokens
single = {'calls': 1, 'in': 3000, 'out': 500}
agent = {'calls': 8, 'in': 8 * 3000, 'out': 8 * 400}
multi = {'calls': 1 + 4 * 8, 'in': 8 * 3000 + 4 * 8 * 3000, 'out': 8 * 400 + 4 * 8 * 400}   # a lead agent plus 4 subagents of 8 steps


def cost(c):
    return c['in'] / 1e6 * PRICE_IN + c['out'] / 1e6 * PRICE_OUT


log(f'== Token cost at an EXAMPLE price of ${PRICE_IN:.2f} / M input and ${PRICE_OUT:.2f} / M output tokens ==')
log(f'{"system":<22} {"calls":>5} {"input tok":>10} {"output tok":>10} {"$ / task":>9} {"$ / 10k tasks":>14}')
rows = {}
for name, c in [('one LLM call', single), ('agent, 8 steps', agent), ('lead + 4 subagents', multi)]:
    d = cost(c)
    rows[name] = {**c, 'usd': d}
    log(f'{name:<22} {c["calls"]:>5} {c["in"]:>10,} {c["out"]:>10,} {d:>9.4f} {10000 * d:>14,.0f}')
log(f'agent / single call: {rows["agent, 8 steps"]["usd"] / rows["one LLM call"]["usd"]:.1f}x the cost, '
    f'{agent["in"] / single["in"]:.0f}x the input tokens')
log(f'multi-agent / single call: {rows["lead + 4 subagents"]["usd"] / rows["one LLM call"]["usd"]:.1f}x the cost')
log('')
LAT = 2.0   # example seconds per model call (round trip); tools add more
log(f'Latency if each model call takes about {LAT:.0f} s (example): one call {LAT:.0f} s, 8 sequential steps {8 * LAT:.0f} s')
save('ch1_math', {'ps': PS, 'ns': NS, 'reliability': rel, 'limits': limits,
                  'price_in': PRICE_IN, 'price_out': PRICE_OUT, 'cost': rows, 'latency_per_call_s': LAT})
