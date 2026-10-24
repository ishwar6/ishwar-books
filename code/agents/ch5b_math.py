"""Chapter 5, Section 5.5: the arithmetic of multi-agent failure. No API calls; every assumption is printed next to its table.
A. Facts surviving a chain of hand-offs: each hop keeps a fact with probability q, so n hops keep q**n of them.
B. A verifier after each hop that catches a share d of the losses and restores them: q' = q + (1 - q) * d.
C. A group chat in which every agent re-reads the whole conversation: input tokens grow with the square of the number of turns.
D. A loop with no hard stop: if the agents decide to stop with probability s per turn, the expected number of turns is 1/s; a cap K bounds it."""
from common import Log, save

log = Log('ch5b_math')
out = {}

log('== A. Facts that survive n hand-offs, if each hop keeps a fact with probability q ==')
log('   assumptions: hops lose facts independently, a lost fact is never recovered later, every fact is equally likely to be lost')
qs, ns = [0.99, 0.95, 0.90, 0.867], [1, 2, 3, 5, 8, 10]
log(f'{"q":>7}' + ''.join(f'{"n=" + str(n):>8}' for n in ns))
out['A'] = {}
for q in qs:
    row = [q ** n for n in ns]
    out['A'][q] = row
    log(f'{q:>7.3f}' + ''.join(f'{v:>8.0%}' for v in row))
log('   q = 0.867 is the per-hop retention implied by our free-text chain (14.7 of 30 facts after 5 hops: (14.7/30)**(1/5))')
log(f'   of 30 facts after 5 hops: q=0.95 keeps {30 * 0.95 ** 5:.1f}, q=0.90 keeps {30 * 0.9 ** 5:.1f}, q=0.867 keeps {30 * 0.867 ** 5:.1f}')

log('\n== B. A verifier after every hop that detects a share d of the losses and gets them restored ==')
log("   q' = q + (1 - q) * d    (assumes a detected loss is always repaired, and the verifier never removes a correct fact)")
out['B'] = {}
for q in [0.90, 0.867]:
    for d in [0.5, 0.8, 0.95]:
        q2 = q + (1 - q) * d
        out['B'][f'{q}_{d}'] = [q2, q2 ** 5]
        log(f'   q={q:.3f}, d={d:.2f}: q\'={q2:.3f}; after 5 hops {q ** 5:.0%} -> {q2 ** 5:.0%}')

log('\n== C. Input tokens when every agent re-reads the whole conversation (a group chat) ==')
log('   assumptions: N agents speak in turn for R rounds; every message is m tokens; each turn reads all earlier messages plus a fixed prompt p')
m, p = 300, 500
log(f'   m = {m} tokens per message, p = {p} tokens of instructions per turn')
log(f'{"agents N":>9}{"rounds R":>9}{"turns":>7}{"group chat":>13}{"briefs only":>13}{"ratio":>7}')
out['C'] = []
for N in [2, 3, 5, 8]:
    for R in [3, 6]:
        T = N * R
        chat = sum(p + (t - 1) * m for t in range(1, T + 1))          # turn t reads t-1 earlier messages
        brief = T * (p + m)                                         # each turn reads only its own brief of about one message
        out['C'].append([N, R, T, chat, brief])
        log(f'{N:>9}{R:>9}{T:>7}{chat:>13,}{brief:>13,}{chat / brief:>7.1f}')
log('   the chat grows as m * T * (T - 1) / 2: doubling the turns roughly quadruples the input tokens')

log('\n== D. Expected turns of a loop that stops only when the agents agree to stop ==')
log('   assumption: on each turn the agents decide to stop with probability s, independently; a hard cap K ends the loop regardless')
out['D'] = []
for s in [0.5, 0.2, 0.1, 0.05]:
    row = [s, 1 / s] + [(1 - (1 - s) ** K) / s for K in [5, 10, 20]]
    out['D'].append(row)
    log(f'   s={s:.2f}: no cap {1 / s:5.1f} turns on average; cap 5 -> {row[2]:.1f}, cap 10 -> {row[3]:.1f}, cap 20 -> {row[4]:.1f}; '
        f'P(still running after 20 turns) = {(1 - s) ** 20:.1%}')
save('ch5b_math', out)
