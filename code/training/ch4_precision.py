"""Chapter 4: what fp32, fp16 and bf16 can and cannot represent (why pretraining uses bf16 with fp32 master weights)."""
import torch
from common import Log, save

log = Log('ch4_precision')
R = {}
log(f'{"format":6s} {"bits":>4s} {"exponent":>8s} {"mantissa":>8s} {"largest":>10s} {"smallest normal":>16s} {"step after 1.0":>15s}')
for name, dt, e, m in [('fp32', torch.float32, 8, 23), ('fp16', torch.float16, 5, 10), ('bf16', torch.bfloat16, 8, 7)]:
    fi = torch.finfo(dt)
    log(f'{name:6s} {fi.bits:4d} {e:8d} {m:8d} {fi.max:10.3g} {fi.tiny:16.3g} {fi.eps:15.3g}')
    R[name] = {'bits': fi.bits, 'exp': e, 'mant': m, 'max': fi.max, 'tiny': fi.tiny, 'eps': fi.eps}
log('')
for v in [3.14159265, 1e-8, 70000.0, 1.0 + 1e-3]:
    row = [f'{name}: {torch.tensor(v, dtype=dt).item():.8g}' for name, dt in [('fp32', torch.float32), ('fp16', torch.float16), ('bf16', torch.bfloat16)]]
    log(f'{v!r:>14} ->  ' + '   '.join(row))
    R[str(v)] = row
log('')
# the update problem: adding a tiny update to a weight of size 1
w, upd = 1.0, 1e-4
for name, dt in [('fp32', torch.float32), ('bf16', torch.bfloat16)]:
    x = torch.tensor(w, dtype=dt)
    for _ in range(1000):
        x = x + torch.tensor(upd, dtype=dt)
    log(f'{name}: start at 1.0, add 1e-4 a thousand times -> {x.item():.6f}  (exact answer 1.1)')
    R[f'acc_{name}'] = x.item()
save('ch4_precision', R)
