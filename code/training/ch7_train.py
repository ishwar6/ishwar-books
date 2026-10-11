"""Chapter 7: train a reward model on UltraFeedback preference pairs.

  python ch7_train.py                                   main run (lr 3e-4), saved to results/ch7_rm/ (Chapter 8 loads it)
  python ch7_train.py --tag lr1 --lr 1e-4               the same with a 3x smaller learning rate (our first attempt)
  python ch7_train.py --tag margin --margin 3 --lr 1e-4 Llama 2 style margin loss ("margin large": 0, 1, 2, 3 by rating gap)
  python ch7_train.py --tag seed1 --seed 1 --n_train 1600 --eval_every 50 --lr 1e-4   another seed and data shard (ensemble)
  python ch7_train.py --tag seed2 --seed 2 --n_train 1600 --eval_every 50 --lr 1e-4
  python ch7_train.py --tag small --seed 3 --n_train 400 --eval_every 10 --lr 1e-4    a reward model trained on 400 pairs

Loss: -log sigmoid(r(x, y_chosen) - r(x, y_rejected) - m), one epoch, AdamW, linear warm-up then linear decay."""
import argparse, json, math, os, random, time
import torch
from common import Log, save
from ch7_common import load_pairs, build_rm, pad, score_ids, bt_loss, margin_of, RM_DIR, CACHE, DEV, tok

ap = argparse.ArgumentParser()
ap.add_argument('--tag', default='main')
ap.add_argument('--n_train', type=int, default=3200)
ap.add_argument('--n_test', type=int, default=600)
ap.add_argument('--margin', type=float, default=0.0)   # 0 = plain Bradley-Terry; 1 = Llama 2 margin small, 3 = margin large
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--bs', type=int, default=8)            # pairs per optimizer step
ap.add_argument('--micro', type=int, default=4)         # pairs per forward pass
ap.add_argument('--eval_every', type=int, default=100)
a = ap.parse_args()

name = f'ch7_train_{a.tag}'
log = Log(name)
t0 = time.time()
# data: the training shard depends on the seed, the test set never changes
train = load_pairs('train_prefs', a.n_train, seed=a.seed)
test = load_pairs('test_prefs', a.n_test, seed=123)
log(f'[{a.tag}] train pairs {len(train)}, test pairs {len(test)}, loaded in {time.time() - t0:.0f}s, device {DEV}')

model = build_rm(seed=a.seed)
n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
log(f'trainable parameters {n_train_params:,} of {n_all:,} ({100 * n_train_params / n_all:.2f}%)')

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps = len(train) // a.bs
warm = max(1, steps // 20)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / (steps - warm))))


def evaluate():
    rc = score_ids(model, [p['ids_c'] for p in test])
    rr = score_ids(model, [p['ids_r'] for p in test])
    acc = sum(c > r for c, r in zip(rc, rr)) / len(test)
    loss = sum(math.log1p(math.exp(-(c - r))) for c, r in zip(rc, rr)) / len(test)
    model.train()
    return acc, loss, rc, rr


acc0, loss0, _, _ = evaluate()
log(f'step    0 | test acc {acc0:.3f} test loss {loss0:.4f} (before training)')
hist = dict(step=[0], train_loss=[None], test_acc=[acc0], test_loss=[loss0], train_acc=[None])
rng = random.Random(a.seed)
order = list(range(len(train)))
rng.shuffle(order)
model.train()
run_loss, run_acc, run_n = 0.0, 0.0, 0
for s in range(steps):
    batch = [train[i] for i in order[s * a.bs:(s + 1) * a.bs]]
    opt.zero_grad()
    for k in range(0, a.bs, a.micro):
        mb = batch[k:k + a.micro]
        ids, att = pad([p['ids_c'] for p in mb] + [p['ids_r'] for p in mb])
        with torch.autocast(DEV, dtype=torch.bfloat16):         # bf16 matmuls: about 30% faster on the Apple GPU
            r = model(input_ids=ids, attention_mask=att).logits[:, 0].float()
        rc, rr = r[:len(mb)], r[len(mb):]
        m = None
        if a.margin:
            # Llama 2 style: a bigger margin for pairs whose ratings differ more
            m = torch.tensor([margin_of(p['sc'] - p['sr'], a.margin) for p in mb], device=DEV)
        loss = bt_loss(rc, rr, m)
        (loss * len(mb) / a.bs).backward()
        run_loss += loss.item() * len(mb)
        run_acc += (rc > rr).float().sum().item()
        run_n += len(mb)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    if (s + 1) % a.eval_every == 0 or s + 1 == steps:
        acc, tl, rcs, rrs = evaluate()
        hist['step'].append(s + 1); hist['train_loss'].append(run_loss / run_n); hist['train_acc'].append(run_acc / run_n)
        hist['test_acc'].append(acc); hist['test_loss'].append(tl)
        log(f'step {s + 1:4d} | train loss {run_loss / run_n:.4f} train acc {run_acc / run_n:.3f} | test acc {acc:.3f} '
            f'test loss {tl:.4f} | {time.time() - t0:.0f}s')
        run_loss, run_acc, run_n = 0.0, 0.0, 0

out = RM_DIR if a.tag == 'main' else os.path.join(CACHE, a.tag)
model.save_pretrained(out)
log(f'saved adapter to {"results/ch7_rm" if a.tag == "main" else out}; total time {time.time() - t0:.0f}s')
save(name, dict(args=vars(a), trainable=n_train_params, total=n_all, hist=hist, steps=steps, minutes=(time.time() - t0) / 60,
                test_rc=rcs, test_rr=rrs))
