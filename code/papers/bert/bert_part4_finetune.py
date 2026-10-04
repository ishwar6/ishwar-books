"""Part 4: a real fine-tune of bert-base-uncased on GLUE SST-2 and MRPC, with the paper's settings.

Paper settings (Section 4.1 and Appendix A.3): batch size 32, 3 epochs, learning rate from {5e-5, 4e-5, 3e-5, 2e-5}.
Like the released run_classifier.py: Adam with weight decay 0.01, linear warmup over the first 10% of steps,
then linear decay to zero, dropout 0.1. We use one learning rate (2e-5) and one seed, no search.
The classifier reads the pooled [CLS] vector (dense + tanh), as the released code does.
Runs on the Apple GPU (MPS). Usage: python bert_part4_finetune.py sst2|mrpc [learning rate]"""
import math, random, sys, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForSequenceClassification
from common import Log, save

TASK = sys.argv[1] if len(sys.argv) > 1 else 'sst2'
LR, BATCH, EPOCHS, MAXLEN, SEED = 2e-5, 32, 3, 128, 0
if len(sys.argv) > 2:                       # optional learning rate from the paper's grid, e.g. 5e-5
    LR = float(sys.argv[2])
NAME = f'part4_{TASK}' if len(sys.argv) <= 2 else f'part4_{TASK}_lr{sys.argv[2]}'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
log = Log(NAME)

ds = load_dataset('nyu-mll/glue', TASK)
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
cols = ('sentence',) if TASK == 'sst2' else ('sentence1', 'sentence2')


def encode(split):
    d = ds[split]
    labels = list(d['label'])
    enc = tok(*[list(d[c]) for c in cols], truncation=True, max_length=MAXLEN)
    return [(enc['input_ids'][i], enc['token_type_ids'][i], labels[i]) for i in range(len(d))]


def batches(data, shuffle, g=None):
    idx = list(range(len(data)))
    if shuffle:
        g.shuffle(idx)
    for k in range(0, len(idx), BATCH):
        part = [data[i] for i in idx[k:k + BATCH]]
        L = max(len(x[0]) for x in part)
        ids = torch.zeros(len(part), L, dtype=torch.long); tt = torch.zeros_like(ids); am = torch.zeros_like(ids)
        for r, (a, t, _) in enumerate(part):
            ids[r, :len(a)] = torch.tensor(a); tt[r, :len(t)] = torch.tensor(t); am[r, :len(a)] = 1
        yield ids.to(dev), tt.to(dev), am.to(dev), torch.tensor([x[2] for x in part]).to(dev)


train, val = encode('train'), encode('validation')
lens = [len(x[0]) for x in train]
log(f'task {TASK}: {len(train)} training examples, {len(val)} dev examples, device {dev}')
log(f'longest training input: {max(lens)} tokens; inputs cut at {MAXLEN}: {sum(l >= MAXLEN for l in lens)}')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(dev)
no_decay = ('bias', 'LayerNorm.weight')
groups = [{'params': [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], 'weight_decay': 0.01},
          {'params': [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], 'weight_decay': 0.0}]
opt = torch.optim.AdamW(groups, lr=LR, betas=(0.9, 0.999), eps=1e-6)
steps = EPOCHS * math.ceil(len(train) / BATCH)
warm = int(0.1 * steps)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: s / warm if s < warm else max(0.0, (steps - s) / (steps - warm)))
log(f'batch {BATCH}, {EPOCHS} epochs = {steps} steps, learning rate {LR}, warmup {warm} steps, weight decay 0.01')


@torch.no_grad()
def evaluate():
    model.eval()
    preds, gold = [], []
    for ids, tt, am, y in batches(val, False):
        preds += model(input_ids=ids, token_type_ids=tt, attention_mask=am).logits.argmax(-1).tolist(); gold += y.tolist()
    model.train()
    p, g = np.array(preds), np.array(gold)
    acc = float((p == g).mean())
    tp, fp, fn = int(((p == 1) & (g == 1)).sum()), int(((p == 1) & (g == 0)).sum()), int(((p == 0) & (g == 1)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn)
    return acc, f1


acc0, f10 = evaluate()
log(f'before fine-tuning (random classifier layer): dev accuracy {acc0:.4f}')
curve, step, t0 = [dict(step=0, dev_acc=acc0)], 0, time.time()
g = random.Random(SEED)
model.train()
every = max(1, steps // 30)
run = []
for ep in range(EPOCHS):
    for ids, tt, am, y in batches(train, True, g):
        loss = model(input_ids=ids, token_type_ids=tt, attention_mask=am, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad()
        step += 1; run.append(loss.item())
        if step % every == 0 or step == steps:
            acc, f1 = evaluate()
            curve.append(dict(step=step, train_loss=float(np.mean(run)), dev_acc=acc, dev_f1=f1))
            run = []
            log(f'step {step:5d}/{steps}  epoch {ep + 1}  train loss {curve[-1]["train_loss"]:.4f}  dev accuracy {acc:.4f}  dev F1 {f1:.4f}  ({time.time() - t0:.0f} s)')
acc, f1 = evaluate()
mins = (time.time() - t0) / 60
log(f'FINAL {TASK} dev accuracy {acc:.4f} ({round(acc * len(val))}/{len(val)}), dev F1 {f1:.4f}, training time {mins:.1f} minutes')
save(NAME, dict(task=TASK, lr=LR, batch=BATCH, epochs=EPOCHS, max_len=MAXLEN, seed=SEED, steps=steps, warmup=warm,
                         n_train=len(train), n_dev=len(val), dev_acc=acc, dev_correct=round(acc * len(val)), dev_f1=f1,
                         minutes=mins, before_acc=acc0, curve=curve))
