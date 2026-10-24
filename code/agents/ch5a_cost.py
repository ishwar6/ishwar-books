"""Chapter 5, Section 5.1: the cost arithmetic of one agent against a lead with m subagents. No API calls: plain arithmetic, with the
size of one supplier page measured from the saved trace of ch5a_single_vs_multi.py (how much the prompt grew after a read_supplier call).
Model: a task needs n tool results of d tokens each; every model call also adds o output tokens to the history. An agent re-sends its
whole context at every model call.
  one agent, one tool call per turn:  input = (n+1)*P + (d+o)*n*(n+1)/2            (n fetch calls and one answer call)
  one agent, b tool calls per turn:   the same with s = n/b calls of b results each   (fewer, larger steps)
  lead + m subagents:                 each subagent does n/m results with its own prompt P_s and brief B; the lead pays a few calls
                                      plus reading m reports of r tokens.
Latency: model calls in a chain; subagents run at the same time, so the lead waits for the slowest one.
Usage: python ch5a_cost.py"""
import json, os, statistics
from common import Log, save

log = Log('ch5a_cost')
HERE = os.path.dirname(os.path.abspath(__file__))


def page_tokens():
    """Median growth of the prompt after a turn whose only tool call was read_supplier, in the one-call-per-turn arm."""
    path = os.path.join(HERE, 'results', 'ch5a_single_vs_multi.json')
    if not os.path.exists(path):
        return None, 0
    runs = json.load(open(path))['runs']
    deltas = []
    for r in runs:
        if r['arm'] != 'seq':
            continue
        tr = r['trace']
        for a, b in zip(tr, tr[1:]):
            if a['tools'] == ['read_supplier']:
                deltas.append(b['prompt_tokens'] - a['prompt_tokens'])        # the page plus the short tool-call message
    deltas = [d for d in deltas if d > 0]
    return (int(statistics.median(deltas)) if deltas else None), len(deltas)


def single(n, P, d, b=1, out=300):
    s = -(-n // b)                                                  # model calls that fetch results, b results per call
    inp = sum(P + i * b * (d + out) for i in range(s + 1))          # s fetch calls + 1 answer call; each re-reads everything before it
    return inp, s + 1


def multi(n, m, P, d, B=500, r=600, lead_calls=3, out=300, b=1):
    k = -(-n // m)                                                  # results per subagent
    sub_inp, sub_calls = single(k, P + B, d, b, out)
    lead_inp = lead_calls * P + m * B + m * r                       # the lead's own calls, its briefs and the reports it reads
    return m * sub_inp + lead_inp, sub_calls, m * sub_calls + lead_calls


if __name__ == '__main__':
    d, nd = page_tokens()
    if d is None:
        d = 700
        log('No saved trace yet: using an assumed page size of 700 tokens.')
    else:
        log(f'Page size measured from the saved trace: median prompt growth after one read_supplier call = {d:,} tokens ({nd} steps).')
    n, P, t_call = 24, 1500, 6.0
    log(f'Assumptions: n = {n} pages to read, a base prompt P = {P:,} tokens, {300} output tokens per call kept in the history,')
    log(f'a 500-token brief and a 600-token report per subagent, 3 calls for the lead, {t_call:.0f} s per model call, tools take no time.')
    log('')
    log(f'{"design":34} {"input tokens":>13} {"calls":>6} {"chain":>6} {"time (s)":>9}')
    rows = []
    for name, (inp, calls, chain) in [
        ('one agent, 1 tool call per turn', (*single(n, P, d, 1), single(n, P, d, 1)[1])),
        ('one agent, 6 tool calls per turn', (*single(n, P, d, 6), single(n, P, d, 6)[1])),
        ('one agent, 24 tool calls per turn', (*single(n, P, d, 24), single(n, P, d, 24)[1])),
    ]:
        rows.append(dict(design=name, input=inp, calls=calls, chain=chain, seconds=chain * t_call))
    for m in (2, 4, 6, 8):
        inp, sub_chain, calls = multi(n, m, P, d)
        rows.append(dict(design=f'lead + {m} subagents (1 call per turn)', input=inp, calls=calls, chain=3 + sub_chain, seconds=(3 + sub_chain) * t_call))
    for m in (4,):
        inp, sub_chain, calls = multi(n, m, P, d, b=6)
        rows.append(dict(design=f'lead + {m} subagents (6 per turn)', input=inp, calls=calls, chain=3 + sub_chain, seconds=(3 + sub_chain) * t_call))
    for r in rows:
        log(f'{r["design"]:34} {r["input"]:13,} {r["calls"]:6} {r["chain"]:6} {r["seconds"]:9.0f}')
    log('')
    base = rows[0]['input']
    q = (d + 300) * n * (n + 1) / 2
    log(f'The re-reading term (d+o)*n*(n+1)/2 alone is {q:,.0f} tokens for one agent with one call per turn, {q / base:.0%} of its input.')
    log('Splitting the reading over m subagents cuts that term by about m; batching tool calls in one agent cuts it by about b.')
    log('Each subagent adds fixed costs (its own prompt, a brief, a report) and the lead adds its own calls.')
    save('ch5a_cost', dict(page_tokens=d, n=n, P=P, t_call=t_call, rows=rows))
