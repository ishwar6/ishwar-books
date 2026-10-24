"""Chapter 6, Section 6.4: calibrate an LLM judge against labels. Real model: gpt-5-mini through ch4_llm.ask (key from a .env outside
the repo, never printed). Data: data/ch6/b/ (run ch6b_data.py first): 80 Northwind support replies with labels set by construction.

Two judge prompts grade every reply PASS or FAIL:
  vague  : "is this a good support reply?" with the policy attached, no criteria;
  rubric : the same inputs plus five binary criteria, one line of reasoning, then the verdict.
On the 'calib' set (40 replies, 20 pass) we build the confusion matrix and measure TPR (judge says PASS when the label is pass) and
TNR (judge says FAIL when the label is fail). On the 'prod' set (40 replies, true pass rate 70%) we compare the judge's raw pass rate
with the corrected estimate theta = (p_obs + TNR - 1) / (TPR + TNR - 1), and a bootstrap 95% interval that resamples both sets.
Writes results/ch6b_judge.json and results/ch6b_judge_stdout.txt."""
import json, os, random
from concurrent.futures import ThreadPoolExecutor
import ch4_llm as llm
from common import Log, save

log = Log('ch6b_judge')
llm.set_log(log)
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ch6', 'b')
POLICY = open(os.path.join(D, 'policy.md')).read()
ORDERS = {o['order_id']: o for o in json.load(open(os.path.join(D, 'orders.json')))}
REPLIES = json.load(open(os.path.join(D, 'replies.json')))
TODAY = '2026-10-20'

CASE = """Today is {today}.
POLICY:
{policy}
ORDER RECORD: {order}
CUSTOMER MESSAGE: {message}
AGENT REPLY: {reply}"""

VAGUE = """You review replies written by a customer-support agent for Northwind Home.
{case}

Is this a good support reply? Answer with JSON only: {{"verdict": "PASS" or "FAIL"}}"""

RUBRIC = """You review replies written by a customer-support agent for Northwind Home. Check the reply against the order record and
the policy. It PASSES only if all five criteria hold; one failure means FAIL.
C1 Facts: every fact about the order (item, status, dates, amounts) matches the order record.
C2 Policy: the decision (refund, return, cancel, replacement) and any amount follow the policy exactly, including fees.
C3 No invented promises: no delivery date, discount, credit or free shipping that the policy does not allow.
C4 Answers the question: the customer's actual question is answered in the first two sentences.
C5 Tone: polite; does not blame the customer.
Compute any amount yourself before you decide.
{case}

Answer with JSON only: {{"failed": [list of failed criteria, e.g. "C2"], "reason": "one sentence", "verdict": "PASS" or "FAIL"}}"""


def judge(r, prompt):
    case = CASE.format(today=TODAY, policy=POLICY, order=json.dumps(ORDERS[r['order_id']]), message=r['message'], reply=r['reply'])
    text, use = llm.ask(prompt.format(case=case), effort='minimal', max_tokens=1500)
    j = llm.parse_json(text) or {}
    v = str(j.get('verdict', '')).upper()
    return {'verdict': 'pass' if v == 'PASS' else 'fail' if v == 'FAIL' else 'fail', 'parsed': v in ('PASS', 'FAIL'),
            'failed': j.get('failed'), 'reason': j.get('reason'), 'use': use}


def confusion(rows):
    tp = sum(r['label'] == 'pass' and r['j'] == 'pass' for r in rows)
    fn = sum(r['label'] == 'pass' and r['j'] == 'fail' for r in rows)
    fp = sum(r['label'] == 'fail' and r['j'] == 'pass' for r in rows)
    tn = sum(r['label'] == 'fail' and r['j'] == 'fail' for r in rows)
    return tp, fn, fp, tn


def rates(rows):
    tp, fn, fp, tn = confusion(rows)
    tpr = tp / (tp + fn) if tp + fn else float('nan')
    tnr = tn / (tn + fp) if tn + fp else float('nan')
    return tpr, tnr


def corrected(p_obs, tpr, tnr):
    den = tpr + tnr - 1
    if den <= 0.05:
        return float('nan')
    return min(1.0, max(0.0, (p_obs + tnr - 1) / den))


def bootstrap(calib, prod, B=2000, seed=6):
    rng = random.Random(seed)
    est = []
    for _ in range(B):
        c = [rng.choice(calib) for _ in calib]
        p = [rng.choice(prod) for _ in prod]
        tpr, tnr = rates(c)
        if tpr != tpr or tnr != tnr:
            continue
        th = corrected(sum(r['j'] == 'pass' for r in p) / len(p), tpr, tnr)
        if th == th:
            est.append(th)
    est.sort()
    return est[int(0.025 * len(est))], est[int(0.975 * len(est)) - 1], len(est)


def main():
    before = llm.snapshot()
    log('$ python ch6b_judge.py')
    log(f'model {llm.MODEL}, reasoning effort minimal; 80 replies x 2 judge prompts = 160 calls')
    out = {}
    for name, prompt in (('vague', VAGUE), ('rubric', RUBRIC)):
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(lambda r: judge(r, prompt), REPLIES))
        rows = [dict(r, j=x['verdict'], parsed=x['parsed'], failed=x['failed'], reason=x['reason']) for r, x in zip(REPLIES, res)]
        calib = [r for r in rows if r['set'] == 'calib']
        prod = [r for r in rows if r['set'] == 'prod']
        tp, fn, fp, tn = confusion(calib)
        tpr, tnr = rates(calib)
        acc = (tp + tn) / len(calib)
        p_obs = sum(r['j'] == 'pass' for r in prod) / len(prod)
        p_true = sum(r['label'] == 'pass' for r in prod) / len(prod)
        th = corrected(p_obs, tpr, tnr)
        lo, hi, nb = bootstrap(calib, prod)
        missed = {}
        for r in calib + prod:
            if r['label'] == 'fail' and r['j'] == 'pass':
                missed[r['defect']] = missed.get(r['defect'], 0) + 1
        log(f'== judge prompt: {name}')
        log(f'calibration set (40 labelled replies, 20 pass / 20 fail)')
        log(f'                 judge PASS   judge FAIL')
        log(f'  label pass      TP = {tp:2d}      FN = {fn:2d}')
        log(f'  label fail      FP = {fp:2d}      TN = {tn:2d}')
        log(f'  accuracy {acc:.1%}   TPR {tpr:.1%}   TNR {tnr:.1%}   unparsed verdicts {sum(not r["parsed"] for r in rows)}')
        log(f'production set (40 replies, true pass rate {p_true:.1%})')
        log(f'  judge raw pass rate {p_obs:.1%}   corrected {th:.1%}   bootstrap 95% interval {lo:.1%} to {hi:.1%} ({nb} resamples)')
        log(f'  false passes by defect (both sets): {missed}')
        out[name] = {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn, 'accuracy': acc, 'tpr': tpr, 'tnr': tnr, 'p_obs': p_obs, 'p_true': p_true,
                     'corrected': th, 'ci': [lo, hi], 'false_pass_by_defect': missed,
                     'rows': [{k: r[k] for k in ('id', 'set', 'label', 'defect', 'j', 'failed', 'reason')} for r in rows]}
    use = llm.diff(before, llm.snapshot())
    out['usage'] = use
    out['cost_usd'] = llm.cost(use)
    out['stub'] = llm.stub_used()
    log(f'== usage: {use["calls"]} calls, {use["input"]} input tokens, {use["output"]} output tokens; cost {out["cost_usd"]:.4f} USD; stub used: {out["stub"]}')
    save('ch6b_judge', out)


if __name__ == '__main__':
    main()
