"""Part 2: what V3's multi-token prediction (MTP) module buys at inference, as arithmetic.

With one MTP module (D = 1), each decoding step proposes the next token (main model) plus one draft token (MTP).
The main model checks the draft in the next pass. If a fraction p of drafts is accepted, a step yields
1 + p tokens on average. The paper reports p = 85% to 90% and 1.8 times tokens per second (2412.19437, p.35).
Also: the extra memory the MTP layer costs, from the parameter counts saved by part2_params.py.
And a toy version of the MTP loss on a 6-token sentence, to show which targets each depth predicts.
"""
import json, os
from common import Log, save

log = Log('part2_mtp')
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, 'results/part2_params.json')))

log('== Which token each depth predicts (T = 6 tokens, D = 1) ==')
toks = ['The', 'cat', 'sat', 'on', 'the', 'mat']
for i in range(len(toks) - 2):
    log(f'  position {i + 1} ({toks[i]:>3s}): main model predicts t{i + 2} = {toks[i + 1]!r:6s} '
        f'MTP module (also given the embedding of t{i + 2}) predicts t{i + 3} = {toks[i + 2]!r}')

log('\n== Tokens per decoding step with one draft token ==')
rows = []
for p in (0.85, 0.90):
    tokens = 1 + p
    rows.append(dict(p=p, tokens=tokens))
    log(f'  acceptance {p:.0%}: 1 + {p:.2f} = {tokens:.2f} tokens per step (upper bound on the speed-up)')
log('  reported by DeepSeek: 1.8x tokens per second; the gap is the cost of running the MTP layer and verifying')

log('\n== What the MTP layer costs in memory ==')
log(f'  unique MTP parameters {P["mtp_unique"] / 1e9:.2f}B = {100 * P["mtp_unique"] / P["total"]:.1f}% of the 671B main model')
log(f'  as stored on Hugging Face (with its own embedding and head copies): {(P["mtp_unique"] + 2 * P["embedding"]) / 1e9:.2f}B')

log('\n== The MTP loss weight over training (paper: lambda 0.3 for the first 10T tokens, 0.1 for the last 4.8T) ==')
for lam in (0.3, 0.1):
    l_main, l_mtp = 2.0, 2.6                     # illustrative loss values, not from the paper
    log(f'  lambda {lam}: total loss = L_main + lambda * L_MTP = {l_main} + {lam} x {l_mtp} = {l_main + lam * l_mtp:.2f}')

save('part2_mtp', dict(rows=rows, mtp_unique=P['mtp_unique'], share=P['mtp_unique'] / P['total']))
