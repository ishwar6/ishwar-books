"""Part 3 (§2.2.2, §2.2.3): the rule-based rewards of DeepSeek-R1-Zero, written two ways and tested on tricky cases.
No model is loaded here (the base-model run is r1_part3_run.py).

  1. accuracy reward for maths: Method 1 our own extract_boxed + normalise + compare; Method 2 the math-verify library;
     a table of tricky answer pairs and where the two disagree.
  2. code reward: run a model-written function against test cases in a subprocess with a timeout.
     Method 1 assert-style tests on a function; Method 2 stdin/stdout tests like a judge.
  3. format reward: Method 1 one regular expression; Method 2 a small tag parser.
  4. reward hacking: a naive reward that a lazy output games, and the strict reward that stops it.
Output: results/r1_part3_stdout.txt and results/part3.json."""
from common import Log, save
from r1_part3_rewards import *   # the reward functions, shown on the page

log = Log('r1_part3')
OUT = {}

# =====================================================================================================================
# 0. Background: how a neural reward model of RLHF learns from one human comparison (Bradley-Terry loss)
# =====================================================================================================================
import math
import torch
import torch.nn.functional as F


def bt_loss(r_chosen, r_rejected):
    """Method 1: loss = -log sigmoid(r_chosen - r_rejected). Small when the chosen answer already scores higher."""
    margin = r_chosen - r_rejected                       # how much higher the preferred answer scores
    p = 1 / (1 + math.exp(-margin))                      # the model's probability that a person prefers 'chosen'
    return -math.log(p), p


log('== 0. reward model loss for one human comparison (Bradley-Terry)')
for rc, rr in [(1.2, 0.4), (0.4, 1.2), (3.0, -3.0)]:
    ours, p = bt_loss(rc, rr)
    lib = -F.logsigmoid(torch.tensor(rc - rr)).item()    # Method 2: the same with PyTorch
    log(f'r(chosen)={rc:+.1f}  r(rejected)={rr:+.1f}  P(chosen preferred)={p:.3f}  loss ours={ours:.3f}  torch={lib:.3f}')
log('')

# =====================================================================================================================
# 1. Accuracy reward for maths
# =====================================================================================================================




log('== 1a. extract the last box: a naive regex vs our brace counter')
for out in ['so the answer is \\boxed{18}.',
            'Thus $\\boxed{\\frac{\\sqrt{3}}{2}}$.',
            'First \\boxed{4}, no wait: \\boxed{\\dfrac{3}{4}}',
            'The answer is \\boxed{\\frac{1}{2}']:
    log(f'{out!r:48} naive regex: {naive_boxed(out)!r:22} ours: {extract_boxed(out)!r}')

PAIRS = [   # (gold answer, the model's final answer inside \boxed{})
    ('\\frac{1}{2}', '0.5'),
    ('\\dfrac{3}{4}', '3/4'),
    ('-\\frac{1}{2}', '-0.5'),
    ('1000', '1,000'),
    ('18', '18 \\text{ dollars}'),
    ('18', '18 dollars'),
    ('3', 'x = 3'),
    ('5', '\\frac{10}{2}'),
    ('1000', '10^{3}'),
    ('\\frac{\\sqrt{2}}{2}', '\\sqrt{2}/2'),
    ('50\\%', '0.5'),
    ('\\{1,2\\}', '\\{2,1\\}'),
    ('\\frac{1}{3}', '0.333'),
    ('2\\pi', '6.28'),
    ('(1,2)', '(2,1)'),
]
log('')
log('== 1b. accuracy reward on tricky pairs (the model output is "The answer is \\boxed{<pred>}.")')
log(f'{"gold":20} {"model answer":22} {"ours":>5} {"lib":>5}  note')
rows, agree = [], 0
for gold, pred in PAIRS:
    out = f'The answer is \\boxed{{{pred}}}.'
    a, b = accuracy_reward_ours(out, gold), accuracy_reward_lib(out, gold)
    note = '' if a == b else 'DISAGREE'
    agree += a == b
    log(f'{gold:20} {pred:22} {a:5.0f} {b:5.0f}  {note}')
    rows.append(dict(gold=gold, pred=pred, ours=a, lib=b))
log(f'the two methods agree on {agree} of {len(PAIRS)} pairs')
OUT['pairs'] = rows

# =====================================================================================================================
# 2. Code reward: run the model's code against tests, in a separate process, with a time limit
# =====================================================================================================================




TESTS = [('digit_sum(0)', 0), ('digit_sum(7)', 7), ('digit_sum(1234)', 10), ('digit_sum(99999)', 45),
         ('digit_sum(10**30)', 1)]
CODES = {
    'correct':        'def digit_sum(n):\n    return sum(int(c) for c in str(n))',
    'drops a digit':     'def digit_sum(n):\n    s = 0\n    while n > 9:\n        s += n % 10\n        n //= 10\n    return s',
    'endless loop':   'def digit_sum(n):\n    s = 0\n    while n >= 0:\n        s += n % 10\n        n //= 10\n    return s',
    'syntax error':   'def digit_sum(n)\n    return sum(map(int, str(n)))',
    'learned the tests': 'def digit_sum(n):\n    return {0: 0, 7: 7, 1234: 10}.get(n, 0)',
    'fakes the marker':  'print("ALL PASS")\nraise SystemExit(0)',
}
log('')
log('== 2. code reward: digit_sum(n), 5 hidden tests, 2 s limit')
OUT['code'] = {}
for name, code in CODES.items():
    r, why = code_reward_asserts(code, TESTS)
    log(f'{name:18} Method 1 (harness): reward {r:.0f}  ({why})')
    OUT['code'][name] = r

JUDGE = [('0\n', '0'), ('7\n', '7'), ('1234\n', '10'), ('99999\n', '45'), ('1' + '0' * 30 + '\n', '1')]
PROGRAMS = {k: v + '\nprint(digit_sum(int(input())))\n' for k, v in CODES.items()}
for name, prog in PROGRAMS.items():
    r, why = code_reward_judge(prog, JUDGE)
    log(f'{name:18} Method 2 (judge)  : reward {r:.0f}  ({why})')
    assert r == OUT['code'][name], 'the two code rewards disagree'
log(f'the two code rewards agree on all {len(PROGRAMS)} programs')
log(f'a grader that trusts a printed "ALL PASS": fakes the marker gets reward '
    f'{code_reward_marker(CODES["fakes the marker"], TESTS):.0f}')

# =====================================================================================================================
# 3. Format reward: <think> ... </think> <answer> ... </answer>
# =====================================================================================================================


FORMAT_CASES = {
    'good':                 '<think>9 eggs at $2 is 18.</think> <answer>18</answer>',
    'no tags':              'She sells 9 eggs, so she makes $18.',
    'no answer tags':       '<think>9 eggs at $2 is 18.</think> The answer is 18.',
    'text before':          'Sure! <think>9 times 2 is 18.</think> <answer>18</answer>',
    'text after':           '<think>9 times 2.</think> <answer>18</answer> User: thanks',
    'empty think':          '<think></think> <answer>18</answer>',
    'two think blocks':     '<think>a guess</think> <think>9 times 2</think> <answer>18</answer>',
    'answer inside think':  '<think>it is <answer>17</answer>, no, 18</think> <answer>18</answer>',
}
log('')
log('== 3. format reward: regex (Method 1) vs parser (Method 2)')
OUT['format'] = {}
for name, out in FORMAT_CASES.items():
    a = format_reward_regex(out)
    b, why = format_reward_parser(out)
    log(f'{name:20} regex {a:.0f}  parser {b:.0f}  ({why}){"   DISAGREE" if a != b else ""}')
    OUT['format'][name] = dict(regex=a, parser=b, why=why)

# =====================================================================================================================
# 4. Reward hacking: a naive rule, a lazy output that games it, and the strict rule
# =====================================================================================================================




honest = '<think>16 - 3 - 4 = 9 eggs; 9 x 2 = 18.</think> <answer>\\boxed{18}</answer>'
spray = '<think>I will list numbers.</think> <answer>' + ' '.join(str(i) for i in range(1001)) + '</answer>'
boxes = '<think>Try all.</think> <answer>' + ''.join(f'\\boxed{{{i}}}' for i in range(30)) + '</answer>'
log('')
log('== 4. reward hacking on the duck-egg question (gold 18)')
for name, out in [('honest answer', honest), ('number spray 0..1000', spray), ('30 boxes 0..29', boxes)]:
    log(f'{name:22} naive reward {naive_reward(out, "18"):.0f}   strict reward {strict_reward(out, "18"):.0f}   '
        f'({len(out)} characters)')
for name in ['empty think', 'two think blocks', 'answer inside think']:
    out = FORMAT_CASES[name].replace('<answer>18</answer>', '<answer>\\boxed{18}</answer>')
    log(f'{name:22} regex format {format_reward_regex(out):.0f}   strict reward {strict_reward(out, "18"):.0f}   '
        f'(the strict reward runs the parser first)')

from datasets import load_dataset
gsm = load_dataset('openai/gsm8k', 'main', split='test')
golds = [r['answer'].split('####')[-1].strip().replace(',', '') for r in gsm]
hit = sum(naive_reward(spray, g) for g in golds)
log(f'number spray on all {len(golds)} GSM8K test questions: naive reward 1 on {hit:.0f} '
    f'({100 * hit / len(golds):.1f}%), strict reward 1 on {sum(strict_reward(spray, g) for g in golds):.0f}')
OUT['hack'] = dict(n=len(golds), naive_hits=hit)
save('part3', OUT)
log('saved results/part3.json')
