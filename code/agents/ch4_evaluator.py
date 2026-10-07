"""Chapter 4, Section 4.6: the evaluator-optimiser loop on a writing-with-constraints task, with the real API (gpt-5-mini,
reasoning_effort=minimal). Eight wordy source paragraphs must each be rewritten to meet five CHECKABLE constraints: 55 to 75 words,
exactly four sentences, no sentence over 20 words, three required keywords present, no banned words. Two loops run from the same start:
  PROGRAMMATIC: generate -> a Python checker lists the violated constraints -> regenerate with the list, up to 4 rounds.
  JUDGE-ONLY:   generate -> a separate model call judges pass or fail per constraint -> regenerate with its verdict, stop when it says pass.
Every draft is also scored by both the checker and the judge, so the chapter can report how often the judge disagrees with the checker
(false passes and false fails), a HELD-OUT fidelity check never fed back to any loop, how many rounds each loop needed, the cost per round, and whether the judge-only loop stopped on drafts
that in fact failed. Token usage and cost come from the API's usage fields."""
import json, re
from common import Log, save
import ch4_llm as llm

log = Log('ch4_evaluator')
llm.set_log(log)
MAX_ROUNDS = 4
BANNED = ['very', 'really', 'basically', 'leverage', 'utilise', 'utilize', 'synergy', 'thing', 'stuff', 'incredibly']

SOURCES = [
    ("Our new onboarding flow is basically a very big improvement over the old one, which was really confusing for a lot of people and had a lot of stuff in it that nobody needed, and we think that by leveraging the new design we will see incredibly good results in terms of how many people finish signing up, although of course we will need to measure this carefully over the coming weeks to be sure that the change is actually doing what we hope it is doing.",
     ['onboarding', 'sign-up', 'measure']),
    ("The migration of the payments service to the new cluster is a thing that we have been planning for a very long time, and it involves moving really quite a lot of traffic, so we are going to do it in stages, starting with internal traffic, then a small percentage of customers, and then everybody, with a rollback plan at each stage in case something goes wrong, which it might.",
     ['payments', 'rollback', 'stages']),
    ("Customer support tickets about delivery delays went up by about forty percent last quarter, which is a very large increase, and when we looked into it we found that most of the increase came from two regions where a courier changed its depot structure, so the fix is really about routing rather than about anything our own teams did wrong, although the communication to customers could basically have been better.",
     ['delivery', 'courier', 'routing']),
    ("The quarterly security review found three medium-severity issues and no high-severity ones, which is incredibly good compared with last year, and the team has already fixed two of the three, with the third one scheduled for the next sprint, and we also want to leverage this moment to remind everyone that dependency updates are a thing that needs doing every month and not every year.",
     ['security', 'dependency', 'sprint']),
    ("Our data warehouse costs have really gone up a lot this year because several teams are running very large queries every hour that scan whole tables when they only need a day of data, and the fix is partly technical, partition filters and query limits, and partly about education, because most of the people writing these queries do not know what a partition is or why it matters.",
     ['warehouse', 'partition', 'cost']),
    ("The mobile app crash rate has basically doubled since the last release, and the stuff we are seeing in the logs points at the new image caching library, which is very memory hungry on older devices, so the plan is to roll back the library this week, then work with the vendor on a fix, then roll forward again once we have really tested it on the low-end devices that our users actually have.",
     ['crash', 'caching', 'devices']),
    ("Hiring for the platform team has been incredibly slow this year, with roles open for an average of five months, and the reasons are a mix of a very competitive market, a long interview process with six stages, and job descriptions that are really written for the people we already have rather than for the people we need, so we are going to cut the process to four stages and rewrite the descriptions.",
     ['hiring', 'interview', 'descriptions']),
    ("The incident on Tuesday was caused by a configuration change that was really meant for the staging environment but was applied to production because the two environments use very similar names and the deployment tool does not ask for confirmation, and the thing we are going to do about it is to rename the environments, add a confirmation step and leverage the existing audit log to alert on production changes made outside of a change window.",
     ['incident', 'configuration', 'confirmation']),
]


# HELD-OUT fidelity check: two key facts per source that a faithful rewrite keeps. Never shown to the generator or the judge, never
# used to stop a loop; applied once, to the final draft, so the loops cannot be tuned to it (unlike C1 to C5, which ARE the feedback).
FACTS = [[r'confus', r'week'], [r'internal', r'customer'], [r'forty|\b40\b', r'two regions|2 regions|two of the regions'],
         [r'three|\b3\b', r'medium'], [r'every hour|hourly|each hour', r'partition filter|query limit'], [r'doubl|twice|2x', r'older|low-end|low end'],
         [r'five months|5 months', r'four|\b4\b'], [r'staging', r'audit log']]


def heldout(text, i):
    low = text.lower()
    return all(re.search(f, low) for f in FACTS[i])


def constraints(keywords):
    return [f'C1: between 55 and 75 words.', f'C2: exactly four sentences.', f'C3: no sentence longer than 20 words.',
            f'C4: contains all of these words: {", ".join(keywords)}.', f'C5: contains none of these words: {", ".join(BANNED)}.']


def sentences(t):
    return [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if s.strip()]


def check(text, keywords):
    """The programmatic evaluator: returns the list of violated constraints with the measured value."""
    v = []
    w = len(text.split())
    if not 55 <= w <= 75:
        v.append(f'C1: {w} words (need 55 to 75)')
    ss = sentences(text)
    if len(ss) != 4:
        v.append(f'C2: {len(ss)} sentences (need exactly 4)')
    long = [len(s.split()) for s in ss if len(s.split()) > 20]
    if long:
        v.append(f'C3: sentence(s) of {long} words (limit 20)')
    low = text.lower()
    missing = [k for k in keywords if k.lower() not in low]
    if missing:
        v.append(f'C4: missing keyword(s) {missing}')
    banned = [b for b in BANNED if re.search(r'\b' + b + r'\b', low)]
    if banned:
        v.append(f'C5: banned word(s) {banned}')
    return v


GEN = 'Rewrite the paragraph below so that it meets ALL of these constraints:\n{cons}\nReply with the rewritten paragraph only.\n\nParagraph: {src}'
REGEN = 'Your rewrite failed these checks:\n- {viol}\nRewrite it again so that it meets ALL the constraints. Reply with the paragraph only.'
JUDGE = ('Check whether the paragraph meets each constraint. Reply with ONE JSON object and nothing else, with keys "C1" to "C5", each "pass" or "fail", '
         'and "notes" (one short sentence).\nConstraints:\n{cons}\n\nParagraph: {text}')


def judge(text, keywords):
    t, u = llm.ask(JUDGE.format(cons='\n'.join(constraints(keywords)), text=text), max_tokens=300)
    d = llm.parse_json(t) or {}
    fails = [c for c in ['C1', 'C2', 'C3', 'C4', 'C5'] if str(d.get(c, 'fail')).lower() != 'pass']
    return fails, d.get('notes', ''), u


def loop(src, keywords, arm, i):
    """arm = 'programmatic' or 'judge'. Returns the per-round record."""
    msgs = [{'role': 'user', 'content': GEN.format(cons='\n'.join(constraints(keywords)), src=src)}]
    rounds = []
    for rnd in range(1, MAX_ROUNDS + 1):
        text, ug = llm.ask(msgs, max_tokens=400)
        viol = check(text, keywords)
        jfails, notes, uj = judge(text, keywords)
        rounds.append({'round': rnd, 'text': text, 'checker_fails': viol, 'judge_fails': jfails, 'judge_notes': notes, 'use_gen': ug, 'use_judge': uj})
        critic_fails = viol if arm == 'programmatic' else jfails
        if not critic_fails:
            break
        feedback = viol if arm == 'programmatic' else [f'{c}: the reviewer marked this constraint as failed. {notes}' for c in jfails]
        msgs += [{'role': 'assistant', 'content': text}, {'role': 'user', 'content': REGEN.format(viol='\n- '.join(feedback))}]
    final = rounds[-1]
    return {'arm': arm, 'rounds': rounds, 'n_rounds': len(rounds), 'stopped_on_pass': not (final['checker_fails'] if arm == 'programmatic' else final['judge_fails']),
            'truly_passes': not final['checker_fails'], 'heldout_ok': heldout(final['text'], i), 'source_heldout_ok': heldout(src, i)}


log(f'== Evaluator-optimiser on {len(SOURCES)} paragraphs, five checkable constraints, up to {MAX_ROUNDS} rounds ({llm.MODEL}, reasoning_effort=minimal) ==')
log('constraints: 55-75 words; exactly 4 sentences; no sentence over 20 words; three required keywords; ten banned words')
log('')
results = {'programmatic': [], 'judge': []}
for i, (src, kw) in enumerate(SOURCES):
    for arm in ['programmatic', 'judge']:
        r = loop(src, kw, arm, i)
        results[arm].append(r)
        trail = ' -> '.join(('pass' if not x['checker_fails'] else ','.join(v.split(':')[0] for v in x['checker_fails'])) for x in r['rounds'])
        log(f'  para {i + 1} {arm:<12} rounds {r["n_rounds"]}  checker trail: {trail:<34} final passes C1-C5: {"yes" if r["truly_passes"] else "NO"}  held-out facts kept: {"yes" if r["heldout_ok"] else "NO"}')

# ---- judge against checker, over every draft produced in both arms
drafts = [x for arm in results.values() for r in arm for x in r['rounds']]
agree = sum(1 for x in drafts if (not x['checker_fails']) == (not x['judge_fails']))
false_pass = sum(1 for x in drafts if x['checker_fails'] and not x['judge_fails'])
false_fail = sum(1 for x in drafts if not x['checker_fails'] and x['judge_fails'])
per_c = {}
for c in ['C1', 'C2', 'C3', 'C4', 'C5']:
    cf = [c in ''.join(x['checker_fails']) for x in drafts]
    jf = [c in x['judge_fails'] for x in drafts]
    per_c[c] = {'agree': sum(1 for a, b in zip(cf, jf) if a == b) / len(drafts), 'checker_fail_rate': sum(cf) / len(drafts), 'judge_fail_rate': sum(jf) / len(drafts)}
log('')
log(f'-- the judge against the programmatic checker, over all {len(drafts)} drafts --')
log(f'overall verdict agreement {agree / len(drafts):.0%}; judge passed a failing draft (false pass) {false_pass} times; judge failed a passing draft (false fail) {false_fail} times')
log(f'{"constraint":<11} {"agreement":>10} {"checker fails":>14} {"judge fails":>12}')
for c, d in per_c.items():
    log(f'{c:<11} {d["agree"]:>10.0%} {d["checker_fail_rate"]:>14.0%} {d["judge_fail_rate"]:>12.0%}')


def summarise(arm):
    rs = results[arm]
    gen = {k: sum(x['use_gen'][k] for r in rs for x in r['rounds']) for k in ['input', 'output', 'reasoning']}
    jud = {k: sum(x['use_judge'][k] for r in rs for x in r['rounds']) for k in ['input', 'output', 'reasoning']}
    n_rounds = sum(r['n_rounds'] for r in rs)
    critic_cost = llm.cost(jud) if arm == 'judge' else 0.0           # the programmatic checker is free; the judge is a model call
    return {'arm': arm, 'rounds_total': n_rounds, 'mean_rounds': n_rounds / len(rs), 'rounds_hist': {k: sum(1 for r in rs if r['n_rounds'] == k) for k in range(1, MAX_ROUNDS + 1)},
            'stopped_on_pass': sum(r['stopped_on_pass'] for r in rs), 'truly_passes': sum(r['truly_passes'] for r in rs), 'n': len(rs),
            'gen_cost_usd': llm.cost(gen), 'critic_cost_usd': critic_cost, 'cost_per_round_usd': (llm.cost(gen) + critic_cost) / n_rounds,
            'first_draft_passes': sum(1 for r in rs if not r['rounds'][0]['checker_fails']), 'heldout_ok': sum(r['heldout_ok'] for r in rs),
            'pass_and_heldout': sum(r['truly_passes'] and r['heldout_ok'] for r in rs)}


summary = {arm: summarise(arm) for arm in results}
log('')
log('-- summary (cost from the API usage fields) --')
log(f'{"critic":<13} {"first draft ok":>14} {"stopped on pass":>16} {"truly passes":>13} {"mean rounds":>12} {"rounds 1/2/3/4":>15} {"gen $":>7} {"critic $":>9} {"$/round":>8} {"held-out ok":>11} {"both":>5}')
for s in summary.values():
    h = '/'.join(str(s['rounds_hist'][k]) for k in range(1, MAX_ROUNDS + 1))
    log(f'{s["arm"]:<13} {s["first_draft_passes"]:>11}/{s["n"]:<2} {s["stopped_on_pass"]:>13}/{s["n"]:<2} {s["truly_passes"]:>10}/{s["n"]:<2} {s["mean_rounds"]:>12.2f} {h:>15} '
        f'{s["gen_cost_usd"]:>7.4f} {s["critic_cost_usd"]:>9.4f} {s["cost_per_round_usd"]:>8.4f} {s["heldout_ok"]:>8}/{s["n"]:<2} {s["pass_and_heldout"]:>3}/{s["n"]:<2}')
log(f'held-out facts present in the SOURCE paragraphs (sanity check of the held-out regexes): {sum(heldout(src, i) for i, (src, _) in enumerate(SOURCES))}/{len(SOURCES)}')
log('reading: "stopped on pass" is what the loop believed; "truly passes" is what the programmatic checker says. Judge errors cut both ways: false passes stop too early, false fails burn rounds.')

U = llm.snapshot()
log('')
log(f'total spend: ${llm.cost(U):.4f} ({U["calls"]} calls, {U["input"]:,} input + {U["output"]:,} output tokens, {U["reasoning"]} hidden reasoning tokens)')
log('stub used: ' + ('YES, results are not real' if llm.stub_used() else 'no, every answer came from the API'))
save('ch4_evaluator', {'model': llm.MODEL, 'stub': llm.stub_used(), 'max_rounds': MAX_ROUNDS, 'summary': summary, 'judge_vs_checker': {
    'drafts': len(drafts), 'agreement': agree / len(drafts), 'false_pass': false_pass, 'false_fail': false_fail, 'per_constraint': per_c},
    'results': results, 'usage': U})
