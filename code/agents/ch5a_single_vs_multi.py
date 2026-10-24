"""Chapter 5, Section 5.1 experiment: one agent against a lead agent with parallel subagents, on Northwind's supplier question.
Both arms use the same model (gpt-5-mini, reasoning effort low), the same task text and the same local tools over the fake catalogue
in data/ch5/a/ (24 supplier pages, a news feed, a shipping table, the buying rules). The single agent has every tool. The lead agent
can only read the rules and the supplier list, and starts subagents that read pages, news and shipping data with their own contexts.
Quality is graded in code against the known answer (gold.json from ch5a_data.py); tokens come from the API's usage fields; time is
wall-clock. Each arm runs REPEATS times. Usage: python ch5a_single_vs_multi.py"""
import json, os, statistics, time
import ch5a_agents as A
from common import Log, save

REPEATS = int(os.environ.get('REPEATS', 5))
ARMS = ('single', 'seq', 'multi')     # one agent; one agent with one tool call per turn; lead + subagents
log = Log('ch5a_single_vs_multi')


def show(arm, i, g, t, secs):
    trap = ', '.join(k for k, v in g['traps'].items() if not v) or 'none'
    log(f'{arm:6} run {i + 1}: pages read {g["pages_read"]:2d}/24, picks {",".join(g["picks"]) or "-":14} gold {g["gold_in_picks"]}/3  rule-breaking {len(g["rule_breaking_picks"])}  '
        f'numbers ok {g["numbers_ok"]}/3  missed traps: {trap}')
    log(f'{"":6}        {t["calls"]:3d} model calls, {t["tool_calls"]:3d} tool calls, input {t["input"]:7,} tok, output {t["output"]:6,} tok, '
        f'peak context {t["peak_context"]:6,} tok, {secs:5.0f} s, ${t["usd"]:.3f}')


if __name__ == '__main__':
    log(f'Model {A.MODEL}, reasoning effort low. Gold picks: {", ".join(A.GOLD["picks"])}. {REPEATS} runs per arm.')
    log('')
    runs, briefs_all = [], []
    for i in range(REPEATS):
        for arm in ARMS:
            m, trace, briefs, subs = A.Meter(), [], [], []
            t0 = time.time()
            ans = (A.run_single(m, trace) if arm == 'single' else A.run_single(m, trace, parallel=False) if arm == 'seq'
                   else A.run_lead(m, trace, briefs, sub_seconds=subs))
            secs = time.time() - t0
            g, t = A.grade(ans), m.total()
            g['pages_read'] = len(A.coverage(trace))
            show(arm, i, g, t, secs)
            if subs:
                log(f'{"":6}        {len(subs)} subagents in parallel, each {min(subs):.0f} to {max(subs):.0f} s; the lead waits for the slowest')
            runs.append(dict(arm=arm, run=i + 1, grade=g, total=t, seconds=secs, by_agent=m.by_agent, answer=ans, trace=trace, briefs=briefs, sub_seconds=subs))
            if briefs:
                briefs_all.append(briefs)
    log('')
    log('Averages over runs (mean; min to max):')
    summary = {}
    for arm in ARMS:
        R = [r for r in runs if r['arm'] == arm]
        f = lambda key: [key(r) for r in R]
        s = {k: f(fn) for k, fn in [('gold', lambda r: r['grade']['gold_in_picks']), ('pages', lambda r: r['grade']['pages_read']), ('numbers', lambda r: r['grade']['numbers_ok']),
                                     ('breaks', lambda r: len(r['grade']['rule_breaking_picks'])), ('input', lambda r: r['total']['input']),
                                     ('output', lambda r: r['total']['output']), ('calls', lambda r: r['total']['calls']),
                                     ('peak', lambda r: r['total']['peak_context']), ('seconds', lambda r: r['seconds']), ('usd', lambda r: r['total']['usd'])]}
        summary[arm] = {k: dict(mean=statistics.mean(v), min=min(v), max=max(v), all=v) for k, v in s.items()}
        S = summary[arm]
        log(f'{arm:6} pages read {S["pages"]["mean"]:.1f}/24 ({S["pages"]["min"]} to {S["pages"]["max"]}), gold picks {S["gold"]["mean"]:.1f}/3 ({S["gold"]["min"]} to {S["gold"]["max"]}), numbers ok {S["numbers"]["mean"]:.1f}/3, '
            f'rule-breaking picks {S["breaks"]["mean"]:.1f}')
        log(f'{"":6} input {S["input"]["mean"]:,.0f} tok ({S["input"]["min"]:,} to {S["input"]["max"]:,}), output {S["output"]["mean"]:,.0f} tok, '
            f'{S["calls"]["mean"]:.0f} calls, peak context {S["peak"]["mean"]:,.0f} tok, {S["seconds"]["mean"]:.0f} s '
            f'({S["seconds"]["min"]:.0f} to {S["seconds"]["max"]:.0f}), ${S["usd"]["mean"]:.3f} per run')
    tot = lambda a: summary[a]['input']['mean'] + summary[a]['output']['mean']
    for a in ('single', 'seq'):
        log(f'multi / {a}: total tokens x{tot("multi") / tot(a):.2f}, time x{summary["multi"]["seconds"]["mean"] / summary[a]["seconds"]["mean"]:.2f}, '
            f'cost x{summary["multi"]["usd"]["mean"] / summary[a]["usd"]["mean"]:.2f}')
    spend = sum(r['total']['usd'] for r in runs)
    log(f'Total API spend for this script: ${spend:.2f}')
    save('ch5a_single_vs_multi', dict(summary=summary, runs=runs, spend=spend, repeats=REPEATS))
