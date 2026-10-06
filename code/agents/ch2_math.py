"""Chapter 2, the arithmetic of context: (1) a budget for one turn's context window, (2) what a 30-step run costs in
input tokens with and without compaction, (3) what one oversized tool result costs when it is re-sent on every later step.
The price is an EXAMPLE used only to make the arithmetic concrete (not a quote from any provider). Token counts are
round planning numbers, not measurements. Writes results/ch2_math.json and results/ch2_math_stdout.txt."""
from common import Log, save

log = Log('ch2_math')
PRICE_IN, PRICE_OUT = 3.00, 15.00            # example: dollars per million input / output tokens

# ---- 1. a context budget for one turn (planning numbers for a mid-sized agent with 20 tools)
budget = {'system instructions': 1500, 'tool definitions (20 x 150)': 3000, 'retrieved memory': 1000,
          'history so far (step 10)': 10800, 'current observation': 800}
total = sum(budget.values())
log('== 1. What one turn of the loop sends to the model (planning numbers, step 10 of a run) ==')
for k, v in budget.items():
    log(f'  {k:<30} {v:>7,} tokens  {100 * v / total:5.1f}%')
log(f'  {"total":<30} {total:>7,} tokens')
log('')

# ---- 2. a 30-step run: the history grows by one model turn (400 tokens) and one observation (800 tokens) per step
STEPS, FIXED, OUT, OBS = 30, 1500 + 3000 + 1000, 400, 800
GROW = OUT + OBS                              # history grows by 1,200 tokens per step
EVERY, SUMMARY, KEEP = 10, 600, 2             # compaction: every 10 steps, replace the history by a 600-token summary, keep the last 2 turns


def run(compact):
    hist, per_step, total_in = 0, [], 0
    for s in range(1, STEPS + 1):
        if compact and s > 1 and (s - 1) % EVERY == 0:
            hist = SUMMARY + KEEP * GROW      # the summary plus the two most recent turns
        ctx = FIXED + hist
        per_step.append(ctx)
        total_in += ctx
        hist += GROW                          # this step's output and observation join the history
    return per_step, total_in


plain, plain_in = run(False)
comp, comp_in = run(True)
out_tokens = STEPS * OUT
cost = lambda i, o: i / 1e6 * PRICE_IN + o / 1e6 * PRICE_OUT
log(f'== 2. A {STEPS}-step run: fixed {FIXED:,} tokens per step, history grows by {GROW:,} tokens per step ==')
log(f'{"step":>4} {"no compaction":>14} {"compaction":>11}')
for s in [1, 5, 10, 11, 20, 21, 30]:
    log(f'{s:>4} {plain[s - 1]:>14,} {comp[s - 1]:>11,}')
log(f'input tokens over the run: {plain_in:>9,} without compaction, {comp_in:>9,} with (every {EVERY} steps, {SUMMARY}-token summary)')
log(f'output tokens over the run: {out_tokens:,} in both cases')
log(f'cost per run at the EXAMPLE price (${PRICE_IN:.0f} / M in, ${PRICE_OUT:.0f} / M out): '
    f'${cost(plain_in, out_tokens):.3f} without, ${cost(comp_in, out_tokens):.3f} with compaction '
    f'({100 * (1 - comp_in / plain_in):.0f}% fewer input tokens, {100 * (1 - cost(comp_in, out_tokens) / cost(plain_in, out_tokens)):.0f}% cheaper)')
log(f'largest context in the run: {max(plain):,} tokens without compaction, {max(comp):,} with')
log('')

# ---- 3. one oversized tool result (a 25,000-token log dump at step 5) is re-sent on every later step
BIG, AT = 25000, 5
resend = BIG * (STEPS - AT + 1)
log(f'== 3. One {BIG:,}-token tool result at step {AT}, re-sent on every later step of the {STEPS}-step run ==')
log(f'  extra input tokens: {resend:,} (that is {resend / plain_in:.1f}x the whole uncompacted run), '
    f'extra cost ${resend / 1e6 * PRICE_IN:.2f} per run at the example price')
log(f'  the same result trimmed to 800 tokens by the tool: {800 * (STEPS - AT + 1):,} extra tokens')
save('ch2_math', {'budget': budget, 'budget_total': total, 'steps': STEPS, 'fixed': FIXED, 'grow': GROW,
                  'compact_every': EVERY, 'summary': SUMMARY, 'keep': KEEP, 'per_step_plain': plain, 'per_step_compact': comp,
                  'in_plain': plain_in, 'in_compact': comp_in, 'out': out_tokens, 'price_in': PRICE_IN, 'price_out': PRICE_OUT,
                  'cost_plain': cost(plain_in, out_tokens), 'cost_compact': cost(comp_in, out_tokens),
                  'big_result': BIG, 'big_at': AT, 'big_resend': resend})
