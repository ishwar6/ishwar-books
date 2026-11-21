"""Part 6: a second, kinder grading of results/part6_eval_gen.jsonl (no model needed).
The base model often gives a boxed answer and then keeps writing (new made-up problems, or a loop), so the last
answer in its text is not its answer. Here each text is cut right after its FIRST \\boxed{...} before grading."""
import json, os
from math_verify import parse, verify
from common import Log, RESULTS

log = Log('part6_rescore')


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


def grade(gold, text):
    try:
        return int(verify(parse('$' + gold + '$'), parse(text)))
    except Exception:
        return 0


log('$ python r1_part6_rescore.py')
rows = [json.loads(l) for l in open(os.path.join(RESULTS, 'part6_eval_gen.jsonl'))]
for tag in ('base', 'distilled'):
    r = [x for x in rows if x['model'] == tag]
    last = sum(x['correct'] for x in r)
    first = sum(grade(x['gold'], first_boxed(x['text'])) for x in r)
    log(f'{tag:9s} graded on the last answer: {last}/{len(r)} = {last / len(r):.3f} | '
        f'graded on the first boxed answer: {first}/{len(r)} = {first / len(r):.3f}')
