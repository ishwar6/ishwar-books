"""Part 1, CUDA and kernels: how many GPU operations one decode step of Qwen2.5-0.5B launches, and what one
launch costs on this machine. Runs on Apple's Metal GPU (MPS), not CUDA; the idea (one launch per operation,
a fixed cost per launch) is the same.
1. Counts the PyTorch operations (aten ops) that run in one decode step, with a dispatch mode.
   View ops (reshape, transpose...) only change how a tensor is read, so they are counted apart.
2. Times many tiny operations in a row to get the cost of one launch, when the arithmetic is nearly zero.
Output: results/p1_kernels_stdout.txt (run with | tee)."""
import statistics, time
from collections import Counter
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import AutoModelForCausalLM, DynamicCache

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
sync = torch.mps.synchronize if DEV == "mps" else (lambda: None)
print(f"device: {DEV}, torch {torch.__version__}")

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16).to(DEV).eval()
V = model.config.vocab_size


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.ops = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops[func.overloadpacket.__name__] += 1
        return func(*args, **(kwargs or {}))


VIEW = {'view', 'reshape', '_unsafe_view', 'transpose', 't', 'expand', 'unsqueeze', 'squeeze', 'slice', 'select',
        'permute', 'alias', 'detach', 'as_strided', 'split', 'chunk', 'unbind', 'lift_fresh', 'empty', 'empty_strided',
        'is_same_size', 'sym_size', 'sym_stride', 'sym_numel', '_local_scalar_dense', 'size', 'stride', 'to'}   # 'to' is a no-op unless it calls _to_copy, which is counted

with torch.inference_mode():
    x = torch.randint(0, V, (1, 512), device=DEV)
    cache = DynamicCache()
    out = model(input_ids=x, past_key_values=cache, logits_to_keep=1)
    tok = out.logits[:, -1:].argmax(-1)
    for _ in range(3):                                   # warm up
        out = model(input_ids=tok, past_key_values=cache, logits_to_keep=1); tok = out.logits[:, -1:].argmax(-1)
    sync()
    with Count() as c:
        out = model(input_ids=tok, past_key_values=cache, logits_to_keep=1)
    sync()
total = sum(c.ops.values())
views = sum(n for k, n in c.ops.items() if k in VIEW)
print(f"one decode step (batch 1, 512-token context): {total} PyTorch operations, {views} of them views or no-op type checks, "
      f"{total - views} that do work on the GPU")
top = [kv for kv in c.ops.most_common(60) if kv[0] not in VIEW][:10]
print('most common working ops:', ', '.join(f'{k} {n}' for k, n in top))
print(f"per layer (24 layers): about {(total - views) / 24:.0f} working ops")

# ---- 2. cost of one launch, when the work is tiny
a = torch.ones(16, device=DEV, dtype=torch.bfloat16)
W = torch.ones(16, 16, device=DEV, dtype=torch.bfloat16)
for name, fn in (("add on 16 numbers", lambda: a + 1), ("16x16 matrix times a vector", lambda: W @ a)):
    for _ in range(200): fn()
    sync()
    per = []
    for _ in range(7):
        sync(); t = time.perf_counter()
        for _ in range(2000): fn()
        sync(); per.append((time.perf_counter() - t) / 2000 * 1e6)
    us = statistics.median(per)
    print(f"tiny op '{name}': {us:.1f} microseconds per op (median of 7 runs of 2,000)")
    if name.startswith('add'):
        launch_us = us
work = total - views
print(f"{work} working ops x {launch_us:.1f} us = {work * launch_us / 1000:.1f} ms of launch cost per decode step, "
      f"if each op costs what a tiny op costs")

# ---- 3. the decode step itself, for comparison (the published value is 10.0 ms)
with torch.inference_mode():
    ts = []
    for _ in range(24):
        sync(); t = time.perf_counter()
        out = model(input_ids=tok, past_key_values=cache, logits_to_keep=1); tok = out.logits[:, -1:].argmax(-1)
        sync(); ts.append(time.perf_counter() - t)
print(f"decode step measured now: {statistics.median(ts[3:]) * 1000:.1f} ms (published Part 1 value: 10.0 ms)")
