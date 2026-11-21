"""Part 8, the run: R1-Zero's recipe with our minir1 package on GSM8K, on one laptop (Apple M-series, mps).
Evaluates 200 held-out GSM8K test questions before and after (greedy, at most 300 new tokens), trains with our own
GRPO, compares our accuracy rule with math_verify and our format parser with an independent regular expression on
every evaluated answer, and saves the final model under
~/.cache/minir1 (Instruct run only, one copy, bfloat16, outside the repository).  Usage: python r1_part8_train.py [model id] [steps]"""
import json, os, re, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from minir1.train import CFG, load_gsm8k, train
from minir1.eval import evaluate
from minir1.rewards import extract_answer

cfg = dict(CFG)
if len(sys.argv) > 1:
    cfg['model'] = sys.argv[1]
if len(sys.argv) > 2:
    cfg['steps'] = int(sys.argv[2]); cfg['minutes'] = float(sys.argv[3]) if len(sys.argv) > 3 else cfg['minutes']
tag = 'base' if cfg['model'].endswith('0.5B') else 'instruct'
log = common.Log(f'r1_part8_train_{tag}')
R = common.RESULTS
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
train_set, test_set = load_gsm8k()
held_out = test_set[:200]
log('config:', json.dumps(cfg))
log(f'GSM8K: {len(train_set)} training questions, {len(held_out)} held-out test questions; device {dev}')


def math_verify_check(outputs, label):
    """Our accuracy rule against math_verify, on the text inside the <answer> tags."""
    from math_verify import parse, verify
    agree, diff = 0, []
    for o in outputs:
        ans = extract_answer(o['text'])
        mv = float(ans is not None and verify(parse(o['gold']), parse(ans)))
        agree += mv == o['acc']
        if mv != o['acc']:
            diff.append((o['gold'], ans, o['acc'], mv))
    log(f'{label}: our accuracy rule and math_verify agree on {agree} of {len(outputs)} answers '
        f'(both read the text inside the last answer tags)')
    for g, a, ours, mv in diff[:6]:
        log(f'   gold {g!r}  answer {a!r:.60}  ours {ours:.0f}  math_verify {mv:.0f}')
    return dict(agree=agree, n=len(outputs), diff=[list(map(str, d)) for d in diff])


# A second, independent way to write the format rule: one regular expression in which neither block may contain a
# tag and each block must hold a non-space character. Our format_reward is a tag parser; the two must agree.
NOT_TAG = r'(?!</?(?:think|answer)>)'                     # what follows is not one of the four tags
BLOCK = rf'(?:{NOT_TAG}.)*?{NOT_TAG}\S(?:{NOT_TAG}.)*?'     # no tag inside, at least one non-space character
FORMAT_RE2 = re.compile(rf'^\s*<think>{BLOCK}</think>\s*<answer>{BLOCK}</answer>\s*$',
                        re.DOTALL)


def format_check(outputs, label):
    agree = sum(float(FORMAT_RE2.match(o['text']) is not None) == o['fmt'] for o in outputs)
    log(f'{label}: our format parser and an independent regular expression agree on {agree} of {len(outputs)} answers')
    return dict(agree=agree, n=len(outputs))


with common.heavy('part8'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model'])
    policy = AutoModelForCausalLM.from_pretrained(cfg['model'], dtype=torch.float32).to(dev)
    t = common.timer()
    before, out_before = evaluate(policy, tok, held_out, cfg['max_new_tokens'])
    log(f'BEFORE training: accuracy {before["acc"]:.3f}  format {before["fmt"]:.3f}  '
        f'mean length {before["length"]:.1f} tokens  ({t():.0f} s)')
    mv_before = math_verify_check(out_before, 'before'); fc_before = format_check(out_before, 'before')
    policy = train(cfg, policy, tok, train_set, log, os.path.join(R, f'r1_part8_log_{tag}.jsonl'),
                   os.path.join(R, f'r1_part8_samples_{tag}.json'))
    t = common.timer()
    after, out_after = evaluate(policy, tok, held_out, cfg['max_new_tokens'])
    log(f'AFTER training: accuracy {after["acc"]:.3f}  format {after["fmt"]:.3f}  '
        f'mean length {after["length"]:.1f} tokens  ({t():.0f} s)')
    mv_after = math_verify_check(out_after, 'after'); fc_after = format_check(out_after, 'after')
    if tag == 'instruct':                         # one checkpoint only, the Instruct run's (it overwrites the old one)
        ck = os.path.expanduser(f'~/.cache/minir1/part8-{tag}')
        policy.to(torch.bfloat16).save_pretrained(ck); tok.save_pretrained(ck)
        log('saved the final model (bfloat16) to ~/.cache/minir1/' + os.path.basename(ck))
common.save(f'r1_part8_eval_{tag}', dict(cfg=cfg, before=before, after=after, mv_before=mv_before, mv_after=mv_after,
                                         fc_before=fc_before, fc_after=fc_after,
                                         out_before=out_before[:40], out_after=out_after[:40]))
