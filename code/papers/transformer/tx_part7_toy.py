"""Part 7: what the small date model (scratch/train_toy.py, saved in scratch/toy.pt) learned.

1. The model and the task.
2. The training curve (from scratch/toy_log.json) and why the loss stops near 0.68 (label smoothing).
3. Outputs on held-out dates, outputs of early checkpoints (wrong), and dates outside the training range.
4. Cross-attention: where each output character looks in the input. Over 2,000 held-out dates: does a digit look at
   the exact digit it copies, does the first month letter look at the month digits?
Writes results/part7_toy_stdout.txt and results/part7_toy.json. CPU, a few seconds."""
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'scratch'))
from common import Log, save
from model import Config, Transformer, greedy_decode, count_parameters
from toy_task import MONTHS as MONTHS_, VOCAB, PAD, BOS, EOS, encode_src, decode_ids, iso, human, split_dates

log = Log('part7_toy')
R = {}
ck = torch.load(os.path.join(HERE, 'scratch', 'toy.pt'))
model = Transformer(Config(**ck['config'])).eval()
model.load_state_dict(ck['state_dict'])
L = json.load(open(os.path.join(HERE, 'scratch', 'toy_log.json')))
cfg = model.cfg

log('== 1. The model')
log(f'task: {ck["task"]}')
log(f'vocabulary: {len(VOCAB)} tokens; d_model {cfg.d_model}, h {cfg.n_heads}, d_ff {cfg.d_ff}, '
    f'layers {cfg.n_encoder_layers} + {cfg.n_decoder_layers}, dropout {cfg.dropout}')
log(f'parameters: {count_parameters(model):,}; file size {os.path.getsize(os.path.join(HERE, "scratch", "toy.pt")) / 1e6:.1f} MB')

log('')
log('== 2. Training (3,000 steps of 128 dates, CPU)')
for row in L['log']:
    if row['step'] in (1, 100, 200, 300, 500, 1000, 2000, 3000):
        log(f'step {row["step"]:>5}  loss {row["loss"]:.3f}  token accuracy {row["token_acc"]:.3f}'
            + (f'  exact match on 500 test dates {row["test_exact"]:.3f}' if 'test_exact' in row else ''))
log(f'training time: {L["seconds"]:.0f} s')
eps, V = 0.1, len(VOCAB)
p_right = 1 - eps + eps / V                                  # PyTorch spreads eps over all V classes
p_other = eps / V
floor = -(p_right * math.log(p_right) + (V - 1) * p_other * math.log(p_other))
log(f'uniform guess over {V} tokens: ln {V} = {math.log(V):.3f}')
log(f'smoothed target: right token {p_right:.4f}, each other token {p_other:.5f}')
log(f'entropy terms: {-p_right * math.log(p_right):.4f} + {-(V - 1) * p_other * math.log(p_other):.4f}')
log(f'lowest possible loss with label smoothing 0.1 and {V} classes: {floor:.3f}')
peak = 128 ** -0.5 * 400 ** -0.5
log(f'Eq. 3 with d_model 128, warmup 400: 128^-0.5 = {128 ** -0.5:.4f}, 400^-0.5 = {400 ** -0.5:.2f}, peak {peak:.5f}')
R['loss_floor'] = floor
R['curve'] = L['log']

log('')
log('== 3. Outputs')
train_dates, test_dates = split_dates()
log(f'dates 1900 to 2099: {len(train_dates) + len(test_dates):,}; training {len(train_dates):,}, held out {len(test_dates):,}')
log(f'held-out test dates (never seen in training): exact match {L["final_exact"] * 100:.1f}% of {L["n_test"]:,}')
for d in test_dates[:6]:
    src = torch.tensor([encode_src(iso(d))])
    log(f'  {iso(d)}  ->  {decode_ids(greedy_decode(model, src, BOS, EOS, 24)[0].tolist())}')
log('early checkpoints, same first three test dates:')
for step, ex in L['snapshots'].items():
    shown = '   '.join(f'{p!r:<20}' for _, _, p in ex[:3])
    log(f'  step {step:>4}: {shown}')
R['snapshots'] = L['snapshots']

log('outside what it was trained on:')
odd = ['2150-07-04', '1850-11-23', '2021-02-30', '2021-13-01', '0001-01-01']
R['odd'] = []
for s in odd:
    out = decode_ids(greedy_decode(model, torch.tensor([encode_src(s)]), BOS, EOS, 24)[0].tolist())
    log(f'  {s}  ->  {out}')
    R['odd'].append([s, out])

log('')
log('== 4. Cross-attention: where does each output character look?')
# Source positions: 2021-03-12 = y y y y - m m - d d </s>  (0..3 year, 5..6 month, 8..9 day, 10 = </s>)


def roles(d):
    """For each output character: the source position it must copy from, or a role name.
    Digits copy one exact source digit. The first month letter needs the month digits (positions 5, 6).
    The other month letters and the spaces follow from what is already written."""
    out = []
    day = str(d.day)
    out += [9] if len(day) == 1 else [8, 9]                  # '1' comes from position 9 of '01'; '12' from 8, 9
    out.append('space')
    out += ['month_first'] + ['month_rest'] * (len(MONTHS_[d.month - 1]) - 1)
    out.append('space')
    out += [0, 1, 2, 3]                                      # the four year digits
    out.append('end')
    return out


@torch.no_grad()
def cross_weights(d):
    """Decode the right answer with teacher forcing and return each decoder layer's cross-attention (h, t, n)."""
    src = torch.tensor([encode_src(iso(d))])
    tgt = torch.tensor([[BOS] + [VOCAB.index(c) for c in human(d)] + [EOS]])
    model(src, tgt[:, :-1])                                  # row k of the weights predicts output character k
    return [layer.cross_attn.last_weights[0] for layer in model.decoder.layers]


H = cfg.n_heads
keys = [(l, hd) for l in range(cfg.n_decoder_layers) for hd in range(H)]
digit_hit = {k: 0 for k in keys}; month_hit = {k: 0 for k in keys}; rest_dash = {k: 0 for k in keys}
n_digit = n_month = n_rest = 0
for d in test_dates:
    ws = cross_weights(d)
    for k, role in enumerate(roles(d)):
        for l, w in enumerate(ws):
            for hd in range(H):
                top = w[hd, k].argmax().item()
                if isinstance(role, int):
                    digit_hit[(l, hd)] += top == role
                elif role == 'month_first':
                    month_hit[(l, hd)] += top in (5, 6)
                elif role == 'month_rest':
                    rest_dash[(l, hd)] += top in (4, 7, 10)  # a dash or </s>: a landmark, not content
        n_digit += isinstance(role, int); n_month += role == 'month_first'; n_rest += role == 'month_rest'
log(f'{len(test_dates):,} held-out dates: {n_digit:,} digits, {n_month:,} first month letters, {n_rest:,} other month letters.')
log('A = digit: the strongest look is at the exact input digit it copies')
log('B = first month letter: the strongest look is at a month digit')
log('C = other month letters: the strongest look is at a dash or </s>')
R['heads'] = {}
for (l, hd) in keys:
    a, b, c = digit_hit[(l, hd)] / n_digit, month_hit[(l, hd)] / n_month, rest_dash[(l, hd)] / n_rest
    log(f'  decoder layer {l + 1}, head {hd}:  A {a * 100:5.1f}%   B {b * 100:5.1f}%   C {c * 100:5.1f}%')
    R['heads'][f'{l + 1}.{hd}'] = [a, b, c]

# One example for the picture
ex = test_dates[0]
ws = cross_weights(ex)
R['example'] = {'src': list(iso(ex)) + ['</s>'], 'tgt': list(human(ex)) + ['</s>'],
                'weights': [w.tolist() for w in ws]}             # [layer][head][t][n]
log(f'example for the attention map: {iso(ex)} -> {human(ex)}')
save('part7_toy', R)
