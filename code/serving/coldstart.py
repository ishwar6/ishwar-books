"""Cold start, measured: how long until a fresh process can serve its first token?

Each trial is a NEW Python process (so imports and GPU set-up are really cold) that times:
  import        import torch and transformers
  read+build    from_pretrained: read the safetensors file and build the model on the CPU
  to GPU        copy the weights to the GPU
  first token   the first forward pass (includes one-time kernel set-up)
  next pass     a second forward pass, for comparison
The model file is likely in the operating system's page cache after the first trial, so 'read+build' is a
best case. The disk itself is measured separately, reading the file with the macOS F_NOCACHE flag set.
Run: python code/serving/coldstart.py  -> results/coldstart.json, results/coldstart_stdout.txt
"""
import fcntl, json, os, subprocess, sys, time
from pathlib import Path
from common import Log

CHILD = r'''
import time, json; t0 = time.perf_counter()
import torch, transformers; t1 = time.perf_counter()
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16); t2 = time.perf_counter()
m = m.to("mps").eval(); torch.mps.synchronize(); t3 = time.perf_counter()
x = torch.randint(0, 1000, (1, 32), device="mps")
with torch.inference_mode(): m(x); torch.mps.synchronize(); t4 = time.perf_counter(); m(x); torch.mps.synchronize(); t5 = time.perf_counter()
print(json.dumps({"import": t1 - t0, "read_build": t2 - t1, "to_gpu": t3 - t2, "first_pass": t4 - t3, "next_pass": t5 - t4}))
'''

log = Log('coldstart')
snap = next((Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots').iterdir())
st = (snap / 'model.safetensors').resolve()
size = st.stat().st_size
log(f'model file: {st.name}, {size / 1e9:.3f} GB')


def read_nocache(path, block=8 << 20):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, 48, 1)                       # F_NOCACHE = 48 on macOS: do not serve from or fill the page cache
    t = time.perf_counter(); n = 0
    while (b := os.read(fd, block)):
        n += len(b)
    os.close(fd)
    return n / (time.perf_counter() - t)

def fresh_file(path, size=2 << 30, block=64 << 20):
    """Write random bytes with F_NOCACHE, so the file is on the SSD but not in memory."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    fcntl.fcntl(fd, 48, 1)
    buf = os.urandom(block)
    for _ in range(size // block):
        os.write(fd, buf)
    os.fsync(fd); os.close(fd)

import tempfile
bws = []
with tempfile.TemporaryDirectory() as tmp:
    for k in range(3):
        f = os.path.join(tmp, f'blob{k}')
        fresh_file(f)
        bws.append(read_nocache(f))
        os.remove(f)
bw = sorted(bws)[1]
log(f'SSD read of a fresh 2 GiB file, F_NOCACHE: {", ".join(f"{b / 1e9:.2f}" for b in bws)} GB/s (median {bw / 1e9:.2f} GB/s)')
log()
trials = []
log(f'{"trial":>5} {"import":>8} {"read+build":>11} {"to GPU":>8} {"1st pass":>9} {"2nd pass":>9} {"total":>8}')
for k in range(5):
    r = json.loads(subprocess.run([sys.executable, '-c', CHILD], capture_output=True, text=True).stdout.strip().splitlines()[-1])
    r['total'] = r['import'] + r['read_build'] + r['to_gpu'] + r['first_pass']
    trials.append(r)
    log(f'{k + 1:5d} {r["import"]:7.2f}s {r["read_build"]:10.2f}s {r["to_gpu"]:7.2f}s {r["first_pass"]:8.2f}s {r["next_pass"] * 1000:7.0f}ms {r["total"]:7.2f}s')
med = {key: sorted(t[key] for t in trials)[2] for key in trials[0]}
log(f'{"median":>5} {med["import"]:7.2f}s {med["read_build"]:10.2f}s {med["to_gpu"]:7.2f}s {med["first_pass"]:8.2f}s {med["next_pass"] * 1000:7.0f}ms {med["total"]:7.2f}s')

log()
log('EXTRAPOLATION: time just to move the weights, at stated bandwidths (no process start, no warm-up)')
sizes = {'Qwen2.5-0.5B BF16': size, 'Llama 3.1 8B BF16': 16.06e9, 'Llama 3.1 8B FP8': 8.03e9, 'Llama 3.1 70B BF16': 141e9}
links = {f'local disk ({bw / 1e9:.1f} GB/s, measured)': bw, '10 Gbit/s network (1.25 GB/s)': 1.25e9,
         '100 Gbit/s network (12.5 GB/s)': 12.5e9}
ext = []
log(f'{"model":>20} ' + ' '.join(f'{k:>32}' for k in links))
for name, s in sizes.items():
    row = {'model': name, 'bytes': s, **{k: s / v for k, v in links.items()}}
    ext.append(row)
    log(f'{name:>20} ' + ' '.join(f'{s / v:31.1f}s' for v in links.values()))
log.save({'file_bytes': size, 'disk_bw': bws, 'trials': trials, 'median': med, 'extrapolation': ext})
