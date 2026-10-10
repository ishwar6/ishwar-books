"""Chapter 4 hands-on: pretrain a 12M-parameter GPT on about 12M tokens of TinyStories on the Apple GPU.
Logs the loss curve, the learning rate, the gradient norm, the validation loss, and samples at several checkpoints."""
import time, torch
from common import Log, save
from ch4_gpt import GPT, get_tokenizer, get_tokens, batch, lr_at

log = Log('ch4_pretrain')
dev = 'mps'
B, T = 32, 256                      # 32 windows of 256 tokens = 8,192 tokens per step
STEPS, WARMUP, PEAK = 1500, 100, 1e-3
SAMPLE_AT = [0, 50, 150, 300, 600, 1000, 1500]

tok = get_tokenizer(4096)
train, val = get_tokens('train', tok), get_tokens('val', tok)
eot = tok.token_to_id('<|endoftext|>')
torch.manual_seed(0)
model = GPT(tok.get_vocab_size(), d=384, n_layer=6, n_head=6, ctx=T).to(dev)
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]       # weight matrices get weight decay
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]     # biases and LayerNorm gains do not
opt = torch.optim.AdamW([{'params': decay, 'weight_decay': 0.1}, {'params': no_decay, 'weight_decay': 0.0}],
                        lr=PEAK, betas=(0.9, 0.95))
g = torch.Generator().manual_seed(1)


@torch.no_grad()
def val_loss(n=40):
    model.eval()
    gv = torch.Generator().manual_seed(123)          # the same 40 validation windows every time
    tot = 0.0
    for _ in range(n):
        x, y = batch(val, B, T, gv, dev)
        with torch.autocast('mps', dtype=torch.bfloat16):
            tot += model(x, y)[1].item()
    model.train()
    return tot / n


def sample(prompt='Once upon a time', n=60):
    model.eval()
    torch.manual_seed(42)
    idx = torch.tensor([tok.encode(prompt).ids], device=dev)
    out = tok.decode(model.generate(idx, n, eos=eot)[0].tolist())
    model.train()
    return out.replace('\n', ' ')


log(f'model: {model.n_params(False):,} parameters ({model.n_params():,} non-embedding)')
log(f'data: {len(train):,} training tokens, {STEPS} steps x {B * T:,} tokens = {STEPS * B * T:,} tokens seen')
hist = {'step': [], 'loss': [], 'lr': [], 'gnorm': [], 'val_step': [], 'val': [], 'samples': {}}
t0 = time.time()
for step in range(STEPS + 1):
    if step % 100 == 0 or step == STEPS:
        v = val_loss()
        hist['val_step'].append(step); hist['val'].append(v)
        log(f'step {step:5d}  val loss {v:.3f}  ({time.time() - t0:.0f}s)')
    if step in SAMPLE_AT:
        s = sample()
        hist['samples'][step] = s
        log(f'   sample @ {step}: {s[:160]}')
    if step == STEPS:
        break
    lr = lr_at(step, PEAK, WARMUP, STEPS)
    for gr in opt.param_groups:
        gr['lr'] = lr
    x, y = batch(train, B, T, g, dev)
    with torch.autocast('mps', dtype=torch.bfloat16):
        _, loss = model(x, y)
    loss.backward()
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if step % 10 == 0:
        hist['step'].append(step); hist['loss'].append(loss.item()); hist['lr'].append(lr); hist['gnorm'].append(gnorm.item())
    if step % 100 == 0:
        log(f'step {step:5d}  train loss {loss.item():.3f}  lr {lr:.2e}  grad norm {gnorm.item():.2f}')
hist['seconds'] = time.time() - t0
hist['params'], hist['params_nonemb'] = model.n_params(False), model.n_params()
log(f'done in {hist["seconds"] / 60:.1f} minutes, {STEPS * B * T / hist["seconds"]:,.0f} tokens/s including evaluation')
torch.save(model.state_dict(), '/Users/admin/.cache/tinystories/ch4_tinygpt.pt')
save('ch4_pretrain', hist)
