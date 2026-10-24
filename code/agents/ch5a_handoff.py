"""Chapter 5, Section 5.3 experiment: what is lost in a handoff. The same research subagent (same model, system prompt and tools)
gets the same six suppliers in three ways:
  thin      a one-line brief, the kind Anthropic's lead agent wrote early on ("research the semiconductor shortage")
  brief     a full brief: objective, the rules with every number, output format, tools and limits
  history   the lead's whole conversation so far (the question, the rules it read, the supplier list, its plan) and then the one-line brief
The six suppliers hold the traps of the catalogue: an open recall (S03), a delivery time over the limit (S07), a name that a US recall
resembles (S08), a closed recall in rupees (S09), a delivered-duty-paid price (S12) and a suspended audit (S19).
Grading: a JSON report is read in code; a prose report goes through one extra model call that turns it into the same small JSON
(landed cost, delivery days, verdict per supplier). Code compares that with gold.json. The extractor's tokens are reported separately
and not charged to the condition.
Usage: python ch5a_handoff.py"""
import json, os, statistics, time
import ch5a_agents as A
import ch4_llm as llm
from common import Log, save

REPEATS = int(os.environ.get('REPEATS', 3))
GROUP = ['S03', 'S07', 'S08', 'S09', 'S12', 'S19']
log = Log('ch5a_handoff')

THIN = f'Research suppliers {", ".join(GROUP)} for Northwind Home\'s 2027 kitchenware sourcing.'

FULL = f"""Objective: check six candidate kitchenware suppliers for Northwind Home (a UK shop) and report the facts the lead needs to choose
three suppliers for 2027. Your suppliers: {", ".join(GROUP)}. Do not research any other supplier.

Rules to apply (from Northwind's buying rules; apply them exactly):
1. Use the "reference basket price" on each supplier page.
2. Convert to GBP with these fixed planning rates, not market rates: 1 EUR = 0.85, 1 USD = 0.78, 1 CNY = 0.11, 1 INR = 0.0092,
   1 TRY = 0.022, 1 PLN = 0.20 GBP.
3. Landed cost per basket: if the price is FOB, add the freight per basket for the supplier's port (shipping_lookup), and add 4% of the
   converted price as import duty if the supplier is outside the EU and the UK. If the price is DDP, the converted price is the landed cost.
4. Delivery days = production lead time (supplier page) + transit days for its port (shipping_lookup). Reject if above 60.
5. Reject if the minimum order quantity is above 1,000 units.
6. Search the news for each supplier by name. Reject for an OPEN safety recall or a suspended ethical audit. A closed recall is not a
   reason to reject, but report it as a risk. Check that a news item is about the same company: similar names can be different companies.

Tools: read_supplier for each page, shipping_lookup for each port, search_news with each supplier's name. Use at most 25 tool calls.

Output: JSON only, one object per supplier:
{{"suppliers": [{{"id": "S..", "currency": "...", "basket_price": 0.0, "incoterm": "FOB or DDP", "landed_cost_gbp": 0.00,
"delivery_days": 0, "moq": 0, "verdict": "accept or reject", "reason": "...", "risks": "..."}}]}}"""

PLAN = ('Plan: 24 suppliers is too many for one context, so I will split them into four groups of six and start four subagents in '
        'parallel. Each subagent checks its suppliers against rules 1 to 6; I will rank the accepted suppliers by landed cost and '
        f'choose three. Group 2 is {", ".join(GROUP)}.')


def history():
    """The lead's conversation up to the handoff, as messages a subagent can read (tool calls written out as text)."""
    return [{'role': 'user', 'content': A.TASK},
            {'role': 'assistant', 'content': 'I will read the buying rules first. [tool call: read_requirements]'},
            {'role': 'user', 'content': '[tool result: read_requirements]\n' + A.read_requirements()},
            {'role': 'assistant', 'content': 'Now the list of candidates. [tool call: list_suppliers]'},
            {'role': 'user', 'content': '[tool result: list_suppliers]\n' + A.list_suppliers()},
            {'role': 'assistant', 'content': PLAN}]


EXTRACT = ('Below is a research report about suppliers {ids}. For each supplier, extract what the REPORT states, without adding your own '
           'calculations. Use null when the report does not state a value. verdict is "accept" if the report says the supplier passes the '
           'rules, is recommended or is chosen, "reject" if it says the supplier fails a rule or should not be used, "unclear" if it gives no verdict. Reply with JSON only: '
           '{{"S03": {{"landed_cost_gbp": number or null, "delivery_days": number or null, "verdict": "accept|reject|unclear"}}, ...}}\n\nREPORT:\n{report}')


def read_report(report, ext_meter):
    """What the report says per supplier: {id: {landed_cost_gbp, delivery_days, verdict}}. A JSON report is read in code (the lead's
    format, or a picks/rejected answer); a prose report goes through one extractor call."""
    d = llm.parse_json(report) or {}
    if isinstance(d.get('suppliers'), list):
        return {str(x.get('id', '')).upper(): x for x in d['suppliers'] if isinstance(x, dict)}, 'code'
    if 'picks' in d or 'rejected' in d:
        out = {str(x.get('id', '')).upper(): dict(x, verdict='accept') for x in d.get('picks', []) if isinstance(x, dict)}
        out.update({str(x.get('id', '')).upper(): dict(x, verdict='reject') for x in d.get('rejected', []) if isinstance(x, dict)})
        return out, 'code'
    msg, use, p = A.chat([{'role': 'user', 'content': EXTRACT.format(ids=', '.join(GROUP), report=report[:20000])}], effort='minimal', max_tokens=2000)
    ext_meter.add('extractor', use, p)
    return llm.parse_json(msg.content) or {}, 'extractor'


def grade(report, ext_meter):
    d, how = read_report(report, ext_meter)
    table = {r['id']: r for r in A.GOLD['table']}
    rows = {}
    for sid in GROUP:
        g, e = table[sid], d.get(sid) or {}
        try:
            cost_ok = e.get('landed_cost_gbp') is not None and abs(float(e['landed_cost_gbp']) - g['landed_gbp']) <= 0.01 * g['landed_gbp']
        except (TypeError, ValueError):
            cost_ok = False
        try:
            days_ok = e.get('delivery_days') is not None and int(float(e['delivery_days'])) == g['delivery_days']
        except (TypeError, ValueError):
            days_ok = False
        want = 'accept' if g['eligible'] else 'reject'
        rows[sid] = dict(cost=cost_ok, days=days_ok, verdict=str(e.get('verdict', '')).lower() == want, said={k: e.get(k) for k in ('landed_cost_gbp', 'delivery_days', 'verdict')},
                         eligible=g['eligible'], graded_by=how)
    return rows


def facts(rows):
    """The 12 facts the lead needs from this handoff: a verdict for each of the six suppliers, and for the three that should pass,
    their landed cost and delivery days (the numbers it will rank on). Rejected suppliers' numbers do not change the decision."""
    return sum(r['verdict'] for r in rows.values()) + sum(r['cost'] + r['days'] for r in rows.values() if r['eligible'])


if __name__ == '__main__':
    log(f'Model {A.MODEL}, reasoning effort low. Suppliers handed off: {", ".join(GROUP)}. {REPEATS} runs per condition.')
    log('Known answer: ' + '; '.join(f'{r["id"]} {r["landed_gbp"]:.2f} GBP, {r["delivery_days"]} d, {"accept" if r["eligible"] else "reject"}'
                                     for r in A.GOLD['table'] if r['id'] in GROUP))
    log(f'Size of what is handed over: thin {len(THIN.split())} words, full brief {len(FULL.split())} words, '
        f'history {sum(len(m["content"].split()) for m in history()) + len(THIN.split())} words.')
    log('')
    ext = A.Meter()
    out = []
    for i in range(REPEATS):
        for cond in ('thin', 'brief', 'history'):
            m = A.Meter()
            brief = FULL if cond == 'brief' else THIN
            hist = history() if cond == 'history' else None
            t0 = time.time()
            report, _secs = A.run_subagent('sub', brief, m, history=hist)
            secs = time.time() - t0
            rows = grade(report, ext)
            t = m.total()
            n = {k: sum(r[k] for r in rows.values() if r['eligible'] or k == 'verdict') for k in ('cost', 'days', 'verdict')}
            n['facts'] = facts(rows)
            took_over = 'picks' in (llm.parse_json(report) or {})          # answered the user's whole question instead of reporting
            parsed = bool(llm.parse_json(report)) and 'suppliers' in (llm.parse_json(report) or {})
            log(f'{cond:7} run {i + 1}: facts right {n["facts"]}/12 (verdicts {n["verdict"]}/6, landed cost {n["cost"]}/3, delivery days {n["days"]}/3); '
                f'lead\'s JSON format: {"yes" if parsed else "no"}; acted as the lead: {"YES" if took_over else "no"}')
            wrong = [f'{s}: said {r["said"].get("verdict", "-")}' for s, r in rows.items() if not r['verdict']]
            log(f'{"":7}        {t["calls"]} calls, {t["tool_calls"]} tool calls, input {t["input"]:,} tok, output {t["output"]:,} tok, '
                f'{secs:.0f} s, ${t["usd"]:.4f}' + (f'; wrong verdicts: {", ".join(wrong)}' if wrong else ''))
            out.append(dict(cond=cond, run=i + 1, rows=rows, n=n, total=t, seconds=secs, report=report, json_format=parsed, took_over=took_over))
    log('')
    log('Means over runs:')
    summary = {}
    for cond in ('thin', 'brief', 'history'):
        R = [o for o in out if o['cond'] == cond]
        S = {k: statistics.mean(o['n'][k] for o in R) for k in ('facts', 'cost', 'days', 'verdict')}
        S['facts_all'] = [o['n']['facts'] for o in R]
        S['took_over'] = sum(o['took_over'] for o in R)
        S.update(input=statistics.mean(o['total']['input'] for o in R), output=statistics.mean(o['total']['output'] for o in R),
                 calls=statistics.mean(o['total']['calls'] for o in R), seconds=statistics.mean(o['seconds'] for o in R),
                 usd=statistics.mean(o['total']['usd'] for o in R), json_format=sum(o['json_format'] for o in R))
        summary[cond] = S
        log(f'{cond:7} facts right {S["facts"]:.1f}/12 (runs: {", ".join(map(str, S["facts_all"]))}), verdicts {S["verdict"]:.1f}/6, '
            f'costs {S["cost"]:.1f}/3, days {S["days"]:.1f}/3; JSON format {S["json_format"]}/{len(R)}, acted as lead {S["took_over"]}/{len(R)}')
        log(f'{"":7} input {S["input"]:,.0f} tok, output {S["output"]:,.0f} tok, {S["calls"]:.1f} calls, '
            f'{S["seconds"]:.0f} s, ${S["usd"]:.4f}')
    spend = sum(o['total']['usd'] for o in out) + ext.total()['usd']
    log(f'Extractor (grading only): {ext.total()["input"]:,} input and {ext.total()["output"]:,} output tokens.')
    log(f'Total API spend for this script: ${spend:.2f}')
    save('ch5a_handoff', dict(summary=summary, runs=out, spend=spend, group=GROUP, thin=THIN, full=FULL, plan=PLAN))
