"""Part 3, Appendix A.2 in numbers: batch arithmetic, epochs, the learning-rate schedule, GELU vs ReLU,
and why length 128 is cheaper than 512. Pure arithmetic; no model needed."""
import math
import torch
from common import Log, save

log = Log('part3_schedule')

# batch and epochs
tokens_exact = 256 * 512
paper_tokens = 128_000
steps = 1_000_000
words = 3.3e9
epochs = paper_tokens * steps / words
log(f'batch: 256 sequences x 512 tokens = {tokens_exact:,} tokens (the paper rounds this to 128,000)')
log(f'1,000,000 steps x 128,000 tokens = {paper_tokens * steps:,.0f} tokens')
log(f'divided by the 3.3 billion words of BooksCorpus + Wikipedia = {epochs:.1f} passes  ("approximately 40 epochs")')
log(f'with the exact 131,072 tokens per batch: {tokens_exact * steps / words:.1f} passes')


# learning-rate schedule: linear warmup to 1e-4 over 10,000 steps, then linear decay to 0 at step 1,000,000
def lr_at(step, peak=1e-4, warmup=10_000, total=1_000_000):
    decayed = peak * (1 - step / total)                 # tf.train.polynomial_decay, power 1, end 0
    return peak * step / warmup if step < warmup else decayed


marks = [0, 2_500, 5_000, 10_000, 100_000, 500_000, 900_000, 1_000_000]
log('')
log('learning rate (released optimization.py: linear warmup, then linear decay to 0):')
for s in marks:
    log(f'   step {s:>9,}: {lr_at(s):.3e}')
curve = [(s, lr_at(s)) for s in range(0, 1_000_001, 5_000)] + [(s, lr_at(s)) for s in range(0, 20_001, 500)]
curve = sorted(set(curve))

# GELU vs ReLU
x = torch.linspace(-4, 4, 161)
gelu = torch.nn.functional.gelu(x)                                       # exact: x * Phi(x)
gelu_tanh = torch.nn.functional.gelu(x, approximate='tanh')              # the formula in the released modeling.py
relu = torch.relu(x)
log('')
log(f'GELU: largest gap between the exact form x*Phi(x) and the tanh formula of the released code: {float((gelu - gelu_tanh).abs().max()):.1e}')
for v in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
    g = v * 0.5 * (1 + math.erf(v / math.sqrt(2)))
    log(f'   x = {v:+.1f}: ReLU {max(0.0, v):+.3f}   GELU {g:+.4f}')
minimum = float(gelu.min())
log(f'   lowest GELU value on [-4, 4]: {minimum:.4f} at x = {float(x[gelu.argmin()]):.2f}')

# attention cost at 128 vs 512
log('')
r_seq = (512 / 128) ** 2
log(f'attention scores per sequence: 128^2 = {128**2:,}, 512^2 = {512**2:,} -> {r_seq:.0f}x more for a 4x longer sequence')
log(f'per token: {512 // 128}x more attention work at length 512')

save('part3_schedule', dict(tokens_exact=tokens_exact, epochs=epochs, epochs_exact=tokens_exact * steps / words,
                            lr_marks={str(s): lr_at(s) for s in marks}, lr_curve=curve,
                            gelu=dict(x=x.tolist(), gelu=gelu.tolist(), relu=relu.tolist(), max_gap_tanh=float((gelu - gelu_tanh).abs().max()), min=minimum),
                            attention=dict(per_seq_ratio=r_seq, per_token_ratio=512 // 128)))
