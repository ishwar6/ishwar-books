"""Train a small Transformer (our model.py) to turn "2021-03-12" into "12 March 2021".

Runs on a laptop CPU in a few minutes. Saves the weights to toy.pt and the log to toy_log.json.
    python train_toy.py            # from code/papers/transformer/scratch/
"""
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from model import Config, Transformer, count_parameters, greedy_decode, config_dict
from toy_task import VOCAB, PAD, BOS, EOS, encode_src, encode_tgt, decode_ids, iso, human, split_dates

HERE = os.path.dirname(os.path.abspath(__file__))
torch.manual_seed(0)
random.seed(0)

# A small model: the paper's design, smaller numbers.
cfg = Config(vocab_size=len(VOCAB), d_model=128, n_heads=4, d_ff=512, n_encoder_layers=2, n_decoder_layers=2,
             dropout=0.1, max_len=32, pad_id=PAD)
STEPS = 3000          # training steps
BATCH = 128           # sentences per step
WARMUP = 400          # warmup_steps of Eq. 3 (the paper uses 4000 for 100,000 steps)
SMOOTHING = 0.1       # label smoothing epsilon_ls (§5.4)
SNAPSHOT_STEPS = (1, 100, 200, 300, 500, 1000)


def pad_batch(seqs):
    """List of id lists -> (b, longest) tensor, filled with PAD on the right."""
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (n - len(s)) for s in seqs])


def make_batch(dates):
    src = pad_batch([encode_src(iso(d)) for d in dates])      # (b, 11): 10 characters + </s>
    tgt = pad_batch([encode_tgt(human(d)) for d in dates])    # (b, t+1): <s> ... </s>
    return src, tgt


def lrate(step):
    """Eq. 3: lrate = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5). Step counts from 1."""
    step = max(step, 1)
    return cfg.d_model ** -0.5 * min(step ** -0.5, step * WARMUP ** -1.5)


@torch.no_grad()
def exact_match(model, dates, batch=500):
    """Greedy-decode every date; count outputs that equal the right answer exactly."""
    right, examples = 0, []
    for k in range(0, len(dates), batch):
        part = dates[k:k + batch]
        src, _ = make_batch(part)
        out = greedy_decode(model, src, BOS, EOS, max_len=24)
        for d, ids in zip(part, out.tolist()):
            pred = decode_ids(ids)
            right += pred == human(d)
            examples.append((iso(d), human(d), pred))
    return right / len(dates), examples


def main():
    train_dates, test_dates = split_dates()
    model = Transformer(cfg)
    print(f'vocabulary: {len(VOCAB)} tokens: {VOCAB}')
    print(f'train dates: {len(train_dates)}, held-out test dates: {len(test_dates)}')
    print(f'parameters: {count_parameters(model):,}')

    # Adam with the paper's settings (§5.3): beta1 = 0.9, beta2 = 0.98, eps = 1e-9. lr=1 so LambdaLR gives Eq. 3.
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: lrate(i + 1))

    log, snapshots, t0 = [], {}, time.time()
    for step in range(1, STEPS + 1):
        model.train()                                          # dropout on
        src, tgt = make_batch(random.sample(train_dates, BATCH))
        dec_in = tgt[:, :-1]                                   # "shifted right": <s> 1 2   M a r c h ...
        labels = tgt[:, 1:]                                    # what must come next: 1 2   M a r c h ... </s>
        logits = model(src, dec_in)                            # (b, t, vocab): teacher forcing, all positions at once
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), labels.reshape(-1),
                               ignore_index=PAD, label_smoothing=SMOOTHING)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 50 == 0 or step == 1:
            keep = labels != PAD
            acc = ((logits.argmax(-1) == labels) & keep).sum().item() / keep.sum().item()
            row = {'step': step, 'loss': round(loss.item(), 4), 'token_acc': round(acc, 4),
                   'lr': round(sched.get_last_lr()[0], 6), 'seconds': round(time.time() - t0, 1)}
            if step % 250 == 0 or step == 1:
                model.eval()
                row['test_exact'] = round(exact_match(model, test_dates[:500])[0], 4)
            if step in SNAPSHOT_STEPS:                         # what the model writes for 6 fixed test dates
                snapshots[step] = exact_match(model, test_dates[:6])[1]
            log.append(row)
            print(json.dumps(row))

    model.eval()
    acc, examples = exact_match(model, test_dates)
    wrong = [e for e in examples if e[1] != e[2]]
    print(f'final exact match on {len(test_dates)} held-out dates: {acc:.4f} ({len(wrong)} wrong)')
    torch.save({'config': config_dict(cfg), 'vocab': VOCAB, 'state_dict': model.state_dict(),
                'task': 'ISO date "2021-03-12" -> "12 March 2021"'}, os.path.join(HERE, 'toy.pt'))
    json.dump({'log': log, 'final_exact': acc, 'n_test': len(test_dates), 'right_examples': examples[:20],
               'wrong_examples': wrong[:50], 'snapshots': snapshots, 'seconds': round(time.time() - t0, 1),
               'parameters': count_parameters(model)}, open(os.path.join(HERE, 'toy_log.json'), 'w'), indent=1)
    print(f'saved toy.pt and toy_log.json after {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
