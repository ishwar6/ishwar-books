"""Chapter 1, experiment 3b: break down the token-shift result (reads results/ch1_token_shift.json, no model needed).
1. Shift rate by kind of prompt.  2. What kind of token the shifted ones are.  3. Shift rate on number tokens (facts)."""
import json, re, collections
from common import Log, save

log = Log('ch1_shift_analysis')
TS = json.load(open('results/ch1_token_shift.json'))
pos = TS['strip_all']
prompts = [p['prompt'] for p in TS['per_prompt']]
KIND = ['knowledge'] * 10 + ['how-to, advice'] * 8 + ['writing'] * 6 + ['maths'] * 6 + ['code'] * 4 + ['chat, identity'] * 6
kind_of = {q: KIND[i] for i, q in enumerate(prompts)}

log('shift rate by kind of prompt:')
by_kind = []
for k in dict.fromkeys(KIND):
    v = [p for p in pos if kind_of[p['prompt_text']] == k]
    r = dict(kind=k, n=len(v), unshifted=sum(p['cls'] == 'unshifted' for p in v) / len(v),
             shifted=sum(p['cls'] == 'shifted' for p in v) / len(v), kl=sum(p['kl'] for p in v) / len(v))
    by_kind.append(r)
    log(f'  {k:<16} {r["n"]:>5} tokens  unshifted {r["unshifted"]:.1%}  shifted {r["shifted"]:.1%}  mean KL {r["kl"]:.3f}')


def kind(p):
    s = p['tok']
    if s == '<|im_end|>':
        return 'end-of-turn token'
    if p['t'] == 0:
        return 'first token of the answer'
    if not re.search(r'[A-Za-z0-9]', s):
        return 'punctuation, spacing, line breaks'
    return 'other words'


sh = [p for p in pos if p['cls'] == 'shifted']
c_sh = collections.Counter(kind(p) for p in sh)
c_all = collections.Counter(kind(p) for p in pos)
log(f'\nwhat the {len(sh)} shifted tokens are (share of shifted | share of all tokens | shift rate inside the group):')
kinds = []
for k in ['end-of-turn token', 'first token of the answer', 'punctuation, spacing, line breaks', 'other words']:
    r = dict(kind=k, n_shifted=c_sh[k], share_shifted=c_sh[k] / len(sh), share_all=c_all[k] / len(pos), rate=c_sh[k] / max(1, c_all[k]))
    kinds.append(r)
    log(f'  {k:<34} {c_sh[k]:>4}  {r["share_shifted"]:6.1%} | {r["share_all"]:6.1%} | {r["rate"]:6.1%}')

nums = [p for p in pos if re.fullmatch(r'\s*\d+', p['tok'])]
log(f'\nnumber tokens (digits) in the answers: {len(nums)}; unshifted {sum(p["cls"] == "unshifted" for p in nums) / len(nums):.1%}, '
    f'shifted {sum(p["cls"] == "shifted" for p in nums) / len(nums):.1%}')
first = [p for p in pos if p['t'] == 0]
log('\nfirst answer token, instruct vs base top-1:')
for p in first:
    if p['cls'] != 'unshifted':
        log(f'  {p["prompt_text"][:40]!r:<44} instruct {p["tok"]!r:<12} base {p["base_top"]!r:<10} ({p["cls"]})')
base_first = collections.Counter(p['base_top'] for p in first).most_common(6)
log(f'base model\'s favourite first tokens: {base_first}')
save('ch1_shift_analysis', dict(by_kind=by_kind, kinds=kinds, n_shifted=len(sh), n_numbers=len(nums),
                                numbers_unshifted=sum(p['cls'] == 'unshifted' for p in nums) / len(nums), base_first=base_first))
