"""Chapter 1, experiment 4: a tiny supervised fine-tuning (SFT) run, written out by hand.
Qwen2.5-0.5B (base), 8 chat examples, full fine-tuning with AdamW for 30 steps. The prompt tokens are masked out of the
loss (label -100), so the model is only trained to produce the assistant's answer and the end-of-turn token <|im_end|>.
We track the training loss, the loss on 3 held-out examples, and the probability of <|im_end|> at the end of a held-out answer."""
import math, sys, torch
from common import Log, save
from ch1_models import load, greedy, BASE, DEV

torch.manual_seed(0)
LR = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-5      # python ch1_sft_tiny.py 1e-4  for the second run
NAME = 'ch1_sft_tiny' if LR == 1e-5 else f'ch1_sft_tiny_lr{sys.argv[1]}'
log = Log(NAME)
TRAIN = [
    ('What is the capital of Japan?', 'The capital of Japan is Tokyo.'),
    ('Name a primary colour.', 'Red is a primary colour.'),
    ('What is 6 times 7?', '6 times 7 is 42.'),
    ('Write a one-line greeting for a birthday card.', 'Happy birthday! Wishing you a year full of joy.'),
    ('Give me one tip for staying hydrated.', 'Carry a water bottle and sip from it throughout the day.'),
    ('What do bees make?', 'Bees make honey and beeswax.'),
    ('Translate "thank you" into Spanish.', '"Thank you" in Spanish is "gracias".'),
    ('How many days are in a week?', 'There are 7 days in a week.'),
]
HELDOUT = [
    ('What is the capital of Italy?', 'The capital of Italy is Rome.'),
    ('How many legs does a spider have?', 'A spider has 8 legs.'),
    ('Give me one tip for better sleep.', 'Go to bed at the same time every night.'),
]
STEPS = 30

tok, model = load(BASE)
END = tok.convert_tokens_to_ids('<|im_end|>')
SYSTEM = 'You are a helpful assistant.'


def encode(q, a):
    """input_ids = prompt + answer + <|im_end|>; labels = -100 on the prompt (ignored by the loss), the token ids elsewhere."""
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    p = p['input_ids'] if isinstance(p, dict) else p
    r = tok(a)['input_ids'] + [END]
    return p + r, [-100] * len(p) + r


def batch(pairs):
    enc = [encode(q, a) for q, a in pairs]
    L = max(len(i) for i, _ in enc)
    pad = tok.convert_tokens_to_ids('<|endoftext|>')
    ids = torch.tensor([i + [pad] * (L - len(i)) for i, _ in enc], device=DEV)
    lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in enc], device=DEV)     # padding is ignored too
    att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in enc], device=DEV)
    return ids, lab, att


# show the masking for one example
ids, lab = encode(*TRAIN[0])
log('one training example, token by token (label -100 = not in the loss):')
for i, l in zip(ids, lab):
    log(f'  {tok.decode([i])!r:<16} label {l if l == -100 else "= " + str(l)}')
log(f'{len(ids)} tokens, {sum(l != -100 for l in lab)} of them are trained on\n')

train = batch(TRAIN)
held = batch(HELDOUT)


@torch.no_grad()
def eval_held():
    model.eval()
    out = model(input_ids=held[0], attention_mask=held[2], labels=held[1])
    # P(<|im_end|>) right after each held-out answer: one unpadded forward pass per example
    ps = []
    for q, a in HELDOUT:
        ids, lab = encode(q, a)
        logits = model(torch.tensor([ids[:-1]], device=DEV)).logits[0, -1].float()   # the prediction for the last token, <|im_end|>
        ps.append(torch.softmax(logits, -1)[END].item())
    model.train()
    return out.loss.item(), sum(ps) / len(ps)


def sample(q):
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    p = p['input_ids'] if isinstance(p, dict) else p
    model.eval()
    new, stopped = greedy(tok, model, p, 40)
    model.train()
    return tok.decode(new), stopped


E = model.get_output_embeddings().weight.detach().float()
Eu = torch.nn.functional.normalize(E[151644:], dim=-1)
cos = Eu[1] @ Eu.T                                             # row 151645 = <|im_end|> against rows 151644 ...
log(f'base model output rows from <|im_start|> (151644) to {E.shape[0] - 1}: {E.shape[0] - 151644} rows, '
    f'cosine similarity of <|im_end|> to the others: min {cos[torch.arange(len(cos)) != 1].min().item():.4f}, '
    f'mean {cos[torch.arange(len(cos)) != 1].mean().item():.4f}; to <|endoftext|>: '
    f'{(torch.nn.functional.normalize(E[151643], dim=-1) @ Eu[1]).item():.4f}')
before = {q: sample(q) for q, _ in HELDOUT}
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
model.train()
hist = []
h_loss, h_end = eval_held()
log(f'step  0: train loss    -     held-out loss {h_loss:.3f}   P(<|im_end|>) on held-out {h_end:.4f}')
hist.append(dict(step=0, train=None, held=h_loss, p_end=h_end))
for step in range(1, STEPS + 1):
    ids, lab, att = train
    out = model(input_ids=ids, attention_mask=att, labels=lab)    # forward: logits, then cross-entropy on the unmasked labels
    out.loss.backward()                                           # backward: a gradient for every one of the 494M weights
    opt.step()                                                    # update: move each weight a little against its gradient
    opt.zero_grad()                                               # clear the gradients for the next step
    h_loss, h_end = eval_held()
    hist.append(dict(step=step, train=out.loss.item(), held=h_loss, p_end=h_end))
    if step <= 5 or step % 5 == 0:
        log(f'step {step:>2}: train loss {out.loss.item():.3f}   held-out loss {h_loss:.3f}   P(<|im_end|>) on held-out {h_end:.4f}')

after = {q: sample(q) for q, _ in HELDOUT}
# check: the probability of <|im_end|> right after the model's own generated answer, and the tokens it generated
model.eval()
for q, a_ in HELDOUT:
    ids, _ = encode(q, a_)
    p_ids = ids[:len(ids) - len(tok(a_)['input_ids']) - 1]
    gen, _ = greedy(tok, model, p_ids, 40)
    with torch.no_grad():
        lg = model(torch.tensor([p_ids + gen[:-1]], device=DEV)).logits[0, -1].float()
    pr = torch.softmax(lg, -1)
    log('  top-5 after the answer: ' + ', '.join(f'{tok.decode([i])!r} {pr[i].item():.4f}' for i in pr.topk(5).indices.tolist())
        + f'   (entropy {-(pr * pr.clamp_min(1e-12).log()).sum().item():.2f} nats)')
    mass = pr[151644:].sum().item()            # <|im_start|>, <|im_end|> and every row after them (control tokens, padding)
    log(f'  probability on the {pr.shape[0] - 151644} rows from <|im_start|> onwards: {mass:.3f}; '
        f'<|im_end|> beats the best of them by {(lg[END] - lg[151644:].topk(2).values[1]).item():.4f} logits')
    log(f'check {q!r}: generated {gen[:-1] == tok(a_)["input_ids"]} same tokens as reference; P(<|im_end|>) after own answer {pr[END].item():.4f}, top-1 {tok.decode([pr.argmax().item()])!r}')
for q, _ in HELDOUT:
    log(f'\nheld-out prompt: {q!r}')
    log(f'  before ({"stopped" if before[q][1] else "did not stop"}): {before[q][0]!r}')
    log(f'  after  ({"stopped" if after[q][1] else "did not stop"}): {after[q][0]!r}')

save(NAME, dict(steps=STEPS, lr=LR, n_train=len(TRAIN), hist=hist,
                          before={q: v for q, v in before.items()}, after={q: v for q, v in after.items()}))
