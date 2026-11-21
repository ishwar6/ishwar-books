"""Part 6: regrade the SAVED answers of r1_part6_eval.py (results/part6_eval_gen.jsonl) with one policy, the same
as Part 4's (r1_part4_analyze.py). No model is loaded.

The policy:
1. An answer that hit the 2,048-token limit has no final answer: it counts as wrong.
2. For the distilled model, only the text after </think> is graded (the summary it writes after thinking).
   A distilled answer that never closes </think> counts as wrong.
3. The base model writes no <think> block, so its whole finished text is graded.
4. math-verify reads the final answer (the last \\boxed{}) of that text and compares it with the gold answer.

The first grading in results/part6_eval_stdout.txt read the WHOLE text of every answer, cut off or not, so a box
written in the middle of the thinking could count. That is why the numbers below are lower.
A second, kinder line grades the base model's finished answers at their FIRST box, because the base model often
boxes an answer and then keeps writing (a made-up new problem, for example)."""
import json, os
from math_verify import parse, verify
from common import Log, save, RESULTS

K, MAX_NEW, N_PROBLEMS = 4, 2048, 16
log = Log('part6_regrade')


def final_text(tag, text, ended):
    """The part of an answer that the policy grades, or None if the answer has no final answer."""
    if not ended:
        return None                               # cut off by the token limit
    if tag == 'distilled':
        if '</think>' not in text:
            return None                           # never finished thinking
        return text.split('</think>')[-1]         # the summary after the thinking
    return text


def first_boxed(text):
    """The text up to the end of the first \\boxed{...} (braces counted), or the whole text if there is none."""
    i = text.find('\\boxed{')
    if i < 0:
        return text
    depth, j = 0, i + len('\\boxed')
    while j < len(text):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        j += 1
        if depth == 0:
            return text[:j]
    return text


def verify_text(gold, text):
    if text is None:
        return 0
    try:
        return int(verify(parse('$' + gold + '$'), parse(text)))
    except Exception:
        return 0


def grade_answer(tag, gold, text, ended):
    """1 if the final answer under the policy equals the gold answer, else 0."""
    return verify_text(gold, final_text(tag, text, ended))


def summarise(r, key):
    n = len(r)
    ok = [x for x in r if x[key]]
    by_level = {lv: (sum(x[key] for x in r if x['level'] == lv) / sum(x['level'] == lv for x in r),
                     sum(x['level'] == lv for x in r) // K) for lv in (1, 2, 3)}
    return dict(pass1=len(ok) / n, correct=len(ok), samples=n, solved_any=len({x['uid'] for x in ok}),
                mean_tokens=sum(x['tokens'] for x in r) / n,
                mean_tokens_correct=sum(x['tokens'] for x in ok) / max(1, len(ok)),
                hit_limit=sum(not x['ended'] for x in r), by_level=by_level)


if __name__ == '__main__':
    log('$ python r1_part6_regrade.py')
    rows = [json.loads(l) for l in open(os.path.join(RESULTS, 'part6_eval_gen.jsonl'))]
    log(f'== {len(rows)} saved answers ({N_PROBLEMS} problems x {K} samples x 2 models), max {MAX_NEW} new tokens')
    log('== policy: cut off at the limit = wrong; distilled: grade only the text after </think>, no </think> = wrong')
    S = {}
    for tag in ('base', 'distilled'):
        r = [x for x in rows if x['model'] == tag]
        for x in r:
            x['policy'] = grade_answer(tag, x['gold'], x['text'], x['ended'])
            ft = final_text(tag, x['text'], x['ended'])
            x['first'] = verify_text(x['gold'], None if ft is None else first_boxed(ft))
        s = summarise(r, 'policy')
        S[tag] = dict(model=tag, **s)
        whole = sum(x['correct'] for x in r)
        finished = sum(x['ended'] for x in r)
        closed = sum('</think>' in x['text'] for x in r)
        lv = ', '.join(f'level {k}: {v[0]:.2f}' for k, v in s['by_level'].items())
        log(f'{tag:9s} pass@1 {s["pass1"]:.3f} ({s["correct"]}/{s["samples"]}) | problems solved at least once '
            f'{s["solved_any"]}/{N_PROBLEMS} | {lv}')
        log(f'{"":9s} finished inside the limit {finished}/{len(r)} | closed </think> {closed}/{len(r)} | '
            f'mean tokens {s["mean_tokens"]:.0f} (correct answers only: {s["mean_tokens_correct"]:.0f})')
        log(f'{"":9s} old grading (whole text, cut-off answers included): {whole}/{len(r)} = {whole / len(r):.3f}')
        if tag == 'base':
            first = sum(x['first'] for x in r)
            S[tag]['first_box_correct'] = first
            log(f'{"":9s} kinder: finished answers graded at their first box: {first}/{len(r)} = {first / len(r):.3f}')
    fin_d = [x for x in rows if x['model'] == 'distilled' and x['ended']]
    log(f'distilled answers that finished: {len(fin_d)}, of which correct: {sum(x["policy"] for x in fin_d)}/{len(fin_d)}')
    save('part6_regrade', dict(summary=S, k=K, max_new=MAX_NEW))
