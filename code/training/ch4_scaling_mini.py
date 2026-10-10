"""Chapter 4: a scaling-law experiment on a laptop. Five GPT sizes (about 0.1M to 10.6M non-embedding parameters) are each
trained for 800 steps (6.6M tokens) on TinyStories with the same recipe; the validation loss is logged every 50 steps,
so each run gives a curve of loss against training compute C = 6 N D."""
import time, torch
from common import Log, save
from ch4_gpt import GPT, get_tokenizer, get_tokens, batch, lr_at

log = Log('ch4_scaling_mini')
dev, B, T, STEPS, PEAK = 'mps', 32, 256, 800, 2e-3
tok = get_tokenizer(4096)
train, val = get_tokens('train', tok), get_tokens('val', tok)
SIZES = [(64, 2, 2), (128, 2, 4), (192, 4, 6), (256, 4, 8), (384, 6, 6)]     # (width d, layers, heads)
out = []
for d, L, H in SIZES:
    torch.manual_seed(0)
    m = GPT(tok.get_vocab_size(), d=d, n_layer=L, n_head=H, ctx=T).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=PEAK, betas=(0.9, 0.95), weight_decay=0.1)
    g = torch.Generator().manual_seed(1)
    N, Nall = m.n_params(), m.n_params(False)
    curve = []
    t0 = time.time()
    for step in range(STEPS + 1):
        if step % 50 == 0 and step > 0:
            m.eval(); gv = torch.Generator().manual_seed(123); tot = 0
            with torch.no_grad():
                for _ in range(20):
                    x, y = batch(val, B, T, gv, dev)
                    with torch.autocast('mps', dtype=torch.bfloat16):
                        tot += m(x, y)[1].item()
            m.train()
            D = step * B * T
            curve.append({'step': step, 'D': D, 'C': 6 * Nall * D, 'val': tot / 20})
        if step == STEPS:
            break
        for gr in opt.param_groups:
            gr['lr'] = lr_at(step, PEAK, 50, STEPS)
        x, y = batch(train, B, T, g, dev)
        with torch.autocast('mps', dtype=torch.bfloat16):
            loss = m(x, y)[1]
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
    log(f'd={d:3d} layers={L} heads={H}: {N:>10,} non-embedding ({Nall:>10,} total) params, final val loss {curve[-1]["val"]:.3f}, '
        f'C={curve[-1]["C"]:.2e} FLOPs, {time.time() - t0:.0f}s')
    out.append({'d': d, 'L': L, 'H': H, 'N': N, 'Nall': Nall, 'curve': curve})
save('ch4_scaling_mini', out)
