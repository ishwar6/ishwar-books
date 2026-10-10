"""Chapter 4: train the BPE tokenizer, encode the TinyStories text, and time one training step in fp32 vs bf16 on the Apple GPU."""
import time, torch
from common import Log, save
from ch4_gpt import GPT, get_tokenizer, get_tokens, batch

log = Log('ch4_prep')
tok = get_tokenizer(4096)
tr, va = get_tokens('train', tok), get_tokens('val', tok)
chars = len(open('/Users/admin/.cache/tinystories/TinyStoriesV2-GPT4-train-head60MB.txt').read())
log(f'vocab size: {tok.get_vocab_size()}')
log(f'train tokens: {len(tr):,}   val tokens: {len(va):,}   train characters: {chars:,}   chars/token: {chars / len(tr):.2f}')
enc = tok.encode('Once upon a time, a little girl named Lily found a shiny red ball.')
log('example:', ' | '.join(enc.tokens))
dev = 'mps'
res = {'vocab': tok.get_vocab_size(), 'train_tokens': int(len(tr)), 'val_tokens': int(len(va)), 'chars_per_token': chars / len(tr)}
for dtype in ['fp32', 'bf16']:
    torch.manual_seed(0)
    m = GPT(4096).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(0)
    times = []
    for i in range(25):
        x, y = batch(tr, 32, 256, g, dev)
        t0 = time.time()
        with torch.autocast('mps', dtype=torch.bfloat16, enabled=(dtype == 'bf16')):
            _, loss = m(x, y)
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        times.append(time.time() - t0)
    t = sum(times[5:]) / 20
    log(f'{dtype}: {t * 1000:.0f} ms per step of 32x256 tokens = {32 * 256 / t:,.0f} tokens/s  (params {m.n_params(False):,}, non-embedding {m.n_params():,})')
    res[dtype] = {'ms_per_step': t * 1000, 'tok_per_s': 32 * 256 / t}
    res['params'], res['params_nonemb'] = m.n_params(False), m.n_params()
save('ch4_prep', res)
