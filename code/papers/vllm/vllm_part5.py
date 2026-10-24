"""Part 5 of vLLM, explained: the experiments (§6 to §10), rebuilt as a small simulation on a laptop.

What it shows (every number is printed and saved to results/part5_stdout.txt and results/part5.json):
  1. Table 1: the number of KV cache slots of each server, from the memory for the KV cache and the size of the
     KV cache of one token (real OPT configs). Checked against the paper's 15.7K, 9.7K and 60.1K.
  2. The workloads (Figure 11): prompt and output lengths of real ShareGPT conversations and Alpaca instructions,
     counted with the OPT tokenizer, against the paper's means.
  3. A step-time model of a real server: one decode step and one prompt pass of OPT-125M on the laptop GPU, timed
     for several batch sizes and lengths, and fitted with a straight line in the number of requests and tokens.
  4. A serving simulator (Method 1): requests arrive at random (Poisson) times; FCFS scheduling; five ways to store
     the KV cache in the same budget of 15,728 slots (Table 1, OPT-13B on one A100): FasterTransformer
     (request-level batching), Orca (Max), Orca (Pow2), Orca (Oracle) with a real buddy allocator, and vLLM
     (16-token blocks on demand, preemption by recomputation). It gives normalized latency against request rate
     (Figure 12) and the number of requests that run at once (Figure 13).
  5. Method 2 for the batch size: a closed form from the memory budget and the lengths (memory full: the slots
     held by the running requests add up to the budget), checked against the simulator.
  6. Figure 15: memory saved by sharing the prompt blocks in parallel sampling. Method 1: a block manager with
     reference counts and copy-on-write; Method 2: a closed form. Both on ShareGPT and Alpaca lengths.
  7. Figure 18b: the simulator with block sizes from 1 to 256.
  8. Figure 19a: swapping a KV cache out to CPU memory and back in, block by block, against recomputing it with
     one prompt pass, on the laptop (OPT-125M, 1,024 tokens).
Model: facebook/opt-125m, FP16 on Apple MPS for the timings. The simulator counts time with the fitted step model.
These are simulations and laptop timings, not the paper's A100 results.
Run: python vllm_part5.py   (about 6 minutes on an M-series Mac; it downloads the ShareGPT file, about 670 MB)"""
import json, math, random, statistics
from collections import deque
import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save, timer

log = Log('part5')
SMALL = 'facebook/opt-125m'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(0)
OUT = {}
tok = AutoTokenizer.from_pretrained(SMALL)


def sync():
    if DEV == 'mps':
        torch.mps.synchronize()


def fastest(fn, repeats=7):
    """Run fn several times; return the fastest wall time in seconds (other work on the laptop only adds time)."""
    best = 1e9
    for _ in range(repeats):
        sync(); clock = timer(); fn(); sync()
        best = min(best, clock())
    return best


# ---------------------------------------------------------------- 1. Table 1: KV cache slots of each server
log('== 1. Table 1: KV cache slots = memory for the KV cache / KV cache of one token')
GiB = 2 ** 30


def kv_bytes_per_token(layers, hidden, bytes_per_value=2):
    """A key and a value (2) of `hidden` numbers in every layer, 2 bytes each in FP16."""
    return 2 * layers * hidden * bytes_per_value


cfg = {m: json.load(open(hf_hub_download(f'facebook/opt-{m}', 'config.json'))) for m in ('13b', '66b')}
SERVERS = [   # (model, layers, hidden, memory for the KV cache in GB as Table 1 gives it, Table 1 slots)
    ('OPT-13B', cfg['13b']['num_hidden_layers'], cfg['13b']['hidden_size'], 12, '15.7K'),
    ('OPT-66B', cfg['66b']['num_hidden_layers'], cfg['66b']['hidden_size'], 21, '9.7K'),
    ('OPT-175B', 96, 12288, 264, '60.1K'),   # OPT-175B weights are not on the Hub; 96 layers x 12288 from the OPT paper
]
log(f'{"model":<9} {"layers":>6} {"hidden":>6} {"KB/token":>9} {"KV memory":>9} {"slots (GiB)":>11} {"Table 1":>8}')
TABLE1 = {}
for name, L, H, gb, paper in SERVERS:
    per = kv_bytes_per_token(L, H)
    slots = gb * GiB / per
    TABLE1[name] = {'layers': L, 'hidden': H, 'bytes_per_token': per, 'kv_gb': gb, 'slots': slots, 'paper': paper}
    log(f'{name:<9} {L:>6} {H:>6} {per / 1024:>9,.0f} {gb:>6} GB {slots:>11,.0f} {paper:>8}')
S = int(TABLE1['OPT-13B']['slots'])
log(f'our simulated server: {S:,} slots (OPT-13B on one A100); with 2048-token requests that is {S / 2048:.2f} requests')
log(f'the same {S:,} slots of OPT-125M take {S * kv_bytes_per_token(12, 768) / 1e6:.0f} MB, so a laptop can hold them')

# ---------------------------------------------------------------- 2. the workloads (Figure 11)
log('')
log('== 2. Figure 11: prompt and output lengths (OPT tokenizer)')
rng = random.Random(0)


def ntok(texts):
    return [len(ids) - 1 for ids in tok(texts).input_ids]     # minus OPT's start token </s>


def keep(p, o):
    """Keep a request only if it fits in the model's 2048 positions (the paper does not say how it cut the data)."""
    return p >= 1 and o >= 1 and p + o <= 2048


sg_path = hf_hub_download('anon8231489123/ShareGPT_Vicuna_unfiltered', 'ShareGPT_V3_unfiltered_cleaned_split.json',
                          repo_type='dataset')
sg = [c['conversations'] for c in json.load(open(sg_path))
      if len(c['conversations']) >= 2 and c['conversations'][0]['from'] == 'human' and c['conversations'][1]['from'] == 'gpt']
pick = rng.sample(range(len(sg)), 6000)
prompts, outputs = [sg[i][0]['value'] for i in pick], [sg[i][1]['value'] for i in pick]   # first question, first answer
SHAREGPT = [(p, o) for p, o in zip(ntok(prompts), ntok(outputs)) if keep(p, o)][:4000]
alp = load_dataset('tatsu-lab/alpaca')['train']
pick = rng.sample(range(len(alp)), 6000)
prompts = [alp[i]['instruction'] + ('\n' + alp[i]['input'] if alp[i]['input'] else '') for i in pick]
ALPACA = [(p, o) for p, o in zip(ntok(prompts), ntok([alp[i]['output'] for i in pick])) if keep(p, o)][:4000]
DATA = {'ShareGPT': SHAREGPT, 'Alpaca': ALPACA}
PAPER_MEANS = {'ShareGPT': (161.31, 337.99), 'Alpaca': (19.31, 58.45)}
log(f'{"dataset":<9} {"requests":>8} {"mean prompt":>11} {"mean output":>11} {"paper prompt":>12} {"paper output":>12}')
LENS = {}
for name, reqs in DATA.items():
    mp, mo = statistics.mean(p for p, _ in reqs), statistics.mean(o for _, o in reqs)
    LENS[name] = {'n': len(reqs), 'prompt_mean': mp, 'output_mean': mo,
                  'prompts': [p for p, _ in reqs], 'outputs': [o for _, o in reqs]}
    log(f'{name:<9} {len(reqs):>8} {mp:>11.1f} {mo:>11.1f} {PAPER_MEANS[name][0]:>12.2f} {PAPER_MEANS[name][1]:>12.2f}')
r_in = LENS['ShareGPT']['prompt_mean'] / LENS['Alpaca']['prompt_mean']
r_out = LENS['ShareGPT']['output_mean'] / LENS['Alpaca']['output_mean']
log(f'ShareGPT / Alpaca: prompts {r_in:.1f}x longer, outputs {r_out:.1f}x longer (paper: 8.4x and 5.8x)')
OUT['lengths'] = LENS

# ---------------------------------------------------------------- 3. a step-time model from real timings
log('')
log('== 3. Step-time model: OPT-125M on the laptop GPU (FP16, fastest of 15 runs)')
model = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float16).to(DEV).eval()
rows, ys = [], []
log(f'{"batch":>5} {"context":>7} {"decode step ms":>14}')
for B, ctx in [(1, 64), (1, 512), (1, 1024), (8, 64), (8, 512), (8, 1024), (32, 64), (32, 512), (32, 1024),
               (64, 64), (64, 256), (64, 512), (128, 64), (128, 256)]:
    ids = torch.randint(100, 20000, (B, ctx), device=DEV)
    with torch.no_grad():
        cache = model(ids, use_cache=True).past_key_values
    step_in = torch.randint(100, 20000, (B, 1), device=DEV)

    def step():
        with torch.no_grad():
            model(step_in, past_key_values=cache, use_cache=True)
            for layer in cache.layers:                      # drop the token just added, so every run is the same
                layer.keys, layer.values = layer.keys[:, :, :ctx], layer.values[:, :, :ctx]
    t = fastest(step, 15)
    rows.append([1.0, B, B * ctx]); ys.append(t)
    log(f'{B:>5} {ctx:>7} {t * 1000:>14.2f}')
    del cache
X, y = np.array(rows), np.array(ys)
A_STEP, B_REQ, C_TOK = np.linalg.lstsq(X, y, rcond=None)[0]
err = np.abs(X @ [A_STEP, B_REQ, C_TOK] - y) / y
log(f'fit: decode step = {A_STEP * 1000:.2f} ms + {B_REQ * 1e6:.1f} us per request + {C_TOK * 1e9:.1f} ns per cached token'
    f' (largest error of the fit {100 * err.max():.0f}%)')
prow, pys = [], []
for n in (64, 128, 256, 512, 1024):
    ids = torch.randint(100, 20000, (1, n), device=DEV)
    t = fastest(lambda: model(ids, use_cache=True), 15)
    prow.append([1.0, n]); pys.append(t)
D_PRE, E_PRE = np.linalg.lstsq(np.array(prow), np.array(pys), rcond=None)[0]
log('prompt pass ms for 64, 128, 256, 512, 1024 tokens: ' + ', '.join(f'{v * 1000:.1f}' for v in pys))
log(f'fit: prompt pass = {D_PRE * 1000:.2f} ms + {E_PRE * 1e6:.1f} us per prompt token')
OUT['step'] = {'a': A_STEP, 'b': B_REQ, 'c': C_TOK, 'e': E_PRE, 'max_err': float(err.max()),
               'grid': [[int(r[1]), int(r[2] / r[1]), v * 1000] for r, v in zip(rows, ys)], 'prompt_ms': [v * 1000 for v in pys]}


def step_time(n_decode, cached, prompt_tokens):
    """One iteration: the fixed cost, plus each decoding request, plus each cached token it reads, plus prompt tokens."""
    return A_STEP + B_REQ * n_decode + C_TOK * cached + E_PRE * prompt_tokens


# ---------------------------------------------------------------- 4. the serving simulator
class Buddy:
    """A buddy allocator over `total` slots, as the paper assumes for Orca. Every chunk is a power of two; a free chunk
    merges with its twin (its buddy) when both are free. The budget is split into power-of-two arenas."""
    def __init__(self, total):
        self.free, self.arenas, self.size_of, base = {}, [], {}, 0
        for k in reversed(range(total.bit_length())):
            if total >> k & 1:
                self.arenas.append((base, 1 << k)); self.free.setdefault(1 << k, set()).add(base); base += 1 << k

    def alloc(self, n):
        need = 1 << (n - 1).bit_length()                        # round up to a power of two
        sizes = [s for s, offs in self.free.items() if s >= need and offs]
        if not sizes:
            return None
        s = min(sizes); off = min(self.free[s]); self.free[s].remove(off)
        while s > need:                                        # split; keep the left half, free the right half
            s //= 2; self.free.setdefault(s, set()).add(off + s)
        self.size_of[off] = need
        return off

    def release(self, off):
        s = self.size_of.pop(off)
        base, arena = next(a for a in self.arenas if a[0] <= off < a[0] + a[1])
        while s < arena:
            twin = base + ((off - base) ^ s)
            if twin not in self.free.get(s, ()):
                break
            self.free[s].remove(twin); off = min(off, twin); s *= 2
        self.free.setdefault(s, set()).add(off)


def reserve(system, p, o):
    """Slots that an Orca version books for a request at admission (before the buddy allocator rounds it up)."""
    if system == 'Orca (Max)':
        return 2048                                            # the model's maximum sequence length
    if system == 'Orca (Pow2)':
        return p + (1 << (o - 1).bit_length())                 # output booked up to 2x: 25 -> 32
    return p + o                                               # Oracle: knows the real output length


def simulate(system, reqs, arrive, block=16, cap=256, slots=S):
    """Method 1: run a server iteration by iteration. Returns normalized latency and how many requests ran at once."""
    n = len(reqs)
    P = [p for p, _ in reqs]; O = [o for _, o in reqs]
    gen, finish = [0] * n, [None] * n
    waiting, running, hold = deque(), [], {}
    t, nxt, area, busy_area, busy_time, preempt, iters = 0.0, 0, 0.0, 0.0, 0.0, 0, 0
    paged, buddy = system.startswith('vLLM'), Buddy(slots)
    free = slots // block
    while nxt < n or waiting or running:
        while nxt < n and arrive[nxt] <= t:
            waiting.append(nxt); nxt += 1
        if not waiting and not running:
            t = arrive[nxt]; continue
        if system == 'FasterTransformer':                      # request-level: a batch runs until its longest request ends
            batch = [waiting.popleft() for _ in range(min(len(waiting), slots // 2048))]
            pmax, omax = max(P[i] for i in batch), max(O[i] for i in batch)
            dt = step_time(0, 0, len(batch) * pmax)            # padded prompts in one pass
            dt += sum(step_time(len(batch), len(batch) * (pmax + k), 0) for k in range(1, omax))
            area += len(batch) * dt; iters += omax
            if waiting:
                busy_area += len(batch) * omax; busy_time += omax
            t += dt
            for i in batch:
                finish[i] = t                                  # the batch comes back as a whole
            continue
        new, preempted = [], False
        if paged:                                              # room for one more token in each running request
            k = 0
            while k < len(running):
                i = running[k]
                need = -(-(P[i] + gen[i] + 1) // block) - hold[i]
                while need > free and running[-1] != i:        # preempt the latest arrival: free all its blocks
                    v = running.pop(); free += hold.pop(v); waiting.appendleft(v); preempt += 1; preempted = True
                if need > free:                                # i itself is the latest arrival
                    running.pop(); free += hold.pop(i); waiting.appendleft(i); preempt += 1; preempted = True
                    break
                free -= need; hold[i] += need; k += 1
        while waiting and len(running) < cap and not preempted:   # admit in arrival order while memory allows
            i = waiting[0]
            if paged:
                need = -(-(P[i] + gen[i] + 1) // block)        # prompt (and tokens to recompute) plus the next token
                if need > free:
                    break
                free -= need; hold[i] = need
            else:
                off = buddy.alloc(reserve(system, P[i], O[i]))
                if off is None:
                    break
                hold[i] = off
            waiting.popleft(); running.append(i); new.append(i)
        if not running:
            t = max(t, arrive[nxt]) if nxt < n else t; continue
        newset = set(new)
        prompt = sum(P[i] + gen[i] for i in new)               # a recomputed request reads its own output again
        dec = [i for i in running if i not in newset]
        dt = step_time(len(dec), sum(P[i] + gen[i] for i in dec), prompt)
        area += len(running) * dt; iters += 1
        if waiting:
            busy_area += len(running); busy_time += 1         # per iteration, while requests wait
        t += dt
        still = []
        for i in running:
            gen[i] += 1
            if gen[i] >= O[i]:
                finish[i] = t
                if paged:
                    free += hold.pop(i)
                else:
                    buddy.release(hold.pop(i))
            else:
                still.append(i)
        running = still
    lat = [(finish[i] - arrive[i]) / O[i] for i in range(n)]
    return {'norm_latency': statistics.mean(lat), 'end': t, 'batched_all': area / t,
            'batched_busy': busy_area / busy_time if busy_time else None, 'preemptions': preempt,
            'tokens_per_s': sum(O) / t, 'tokens_per_step': sum(O) / iters}


def poisson(n, rate, seed):
    """Arrival times of n requests: gaps drawn from an exponential distribution with mean 1 / rate."""
    r, t, out = random.Random(seed), 0.0, []
    for _ in range(n):
        t += r.expovariate(rate); out.append(t)
    return out


SYSTEMS = ['FasterTransformer', 'Orca (Max)', 'Orca (Pow2)', 'Orca (Oracle)', 'vLLM']
N_REQ = 1000
TRACE = {name: random.Random(1).sample(reqs, N_REQ) for name, reqs in DATA.items()}
RATES = {'ShareGPT': [0.25, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8],
         'Alpaca': [5, 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 200]}
LIMIT = 0.1                                                    # s/token: "still fast" for our laptop-scale server

log('')
log(f'== 4. Figure 12 rebuilt: normalized latency (s/token) against request rate, {N_REQ} requests per point, '
    f'{S:,} slots')
log('normalized latency = mean over requests of (finish time - arrival time) / output tokens')
CURVES = {}
for ds, reqs in TRACE.items():
    log(f'{ds}:')
    log(f'{"req/s":>6} ' + ' '.join(f'{s:>17}' for s in SYSTEMS))
    CURVES[ds] = {s: [] for s in SYSTEMS}
    for rate in RATES[ds]:
        arrive = poisson(N_REQ, rate, seed=2)
        vals = [simulate(s, reqs, arrive)['norm_latency'] for s in SYSTEMS]
        for s, v in zip(SYSTEMS, vals):
            CURVES[ds][s].append(v)
        log(f'{rate:>6} ' + ' '.join(f'{v:>17.3f}' for v in vals))
OUT['curves'] = CURVES
OUT['rates'] = RATES

log('')
log(f'request rate where normalized latency crosses {LIMIT} s/token (our "same latency" line, linear between points):')


def crossing(rates, vals, limit=LIMIT):
    """The rate at which the curve first rises above the limit, read between the two points around it."""
    if vals[0] > limit:
        return 0.0
    for (r0, v0), (r1, v1) in zip(zip(rates, vals), zip(rates[1:], vals[1:])):
        if v1 > limit:
            return r0 + (r1 - r0) * (limit - v0) / (v1 - v0)
    return float('inf')


CAP = {}
for ds in TRACE:
    CAP[ds] = {s: crossing(RATES[ds], CURVES[ds][s]) for s in SYSTEMS}
    log(f'{ds:<9} ' + ', '.join(f'{s} ' + (f'{CAP[ds][s]:.1f}' if CAP[ds][s] else f'< {RATES[ds][0]}') for s in SYSTEMS))
    v = CAP[ds]['vLLM']
    log(f'{"":<9} vLLM / Orca (Oracle) = {v / CAP[ds]["Orca (Oracle)"]:.1f}x, vLLM / Orca (Pow2) = '
        f'{v / CAP[ds]["Orca (Pow2)"]:.1f}x, vLLM / Orca (Max) = {v / CAP[ds]["Orca (Max)"]:.1f}x')
log('paper, ShareGPT: vLLM sustains 1.7x to 2.7x the rate of Orca (Oracle), 2.7x to 8x that of Orca (Max), '
    'up to 22x that of FasterTransformer')
OUT['capacity'] = CAP

# ---------------------------------------------------------------- 5. Figure 13: how many requests run at once
log('')
log('== 5. Figure 13 rebuilt: requests running at once while the queue is never empty (all arrive at time 0)')


def closed_form_batch(system, reqs, block=16, slots=S):
    """Method 2: when memory is full, the slots held by the running requests add up to the budget. A request is
    running for o steps, so a random running request is picked in proportion to o. Fragmentation is ignored."""
    held = 0
    for p, o in reqs:
        if system == 'vLLM':
            held += sum(block * -(-(p + g) // block) for g in range(1, o + 1))   # blocks grow as it writes
        else:
            held += o * (1 << (reserve(system, p, o) - 1).bit_length())         # one chunk for its whole life
    return slots * sum(o for _, o in reqs) / held


BATCH = {}
for ds, reqs in TRACE.items():
    BATCH[ds] = {}
    log(f'{ds}: {"system":<18} {"simulated":>9} {"closed form":>11} {"tokens/step":>11} {"tokens/s":>9} {"preemptions":>11}')
    for s in SYSTEMS[1:]:
        r = simulate(s, reqs, [0.0] * N_REQ)
        cf = closed_form_batch(s, reqs)
        BATCH[ds][s] = {'sim': r['batched_busy'], 'closed': cf, 'tokens_per_s': r['tokens_per_s'],
                        'tokens_per_step': r['tokens_per_step'], 'preemptions': r['preemptions']}
        log(f'{"":<9} {s:<18} {r["batched_busy"]:>9.2f} {cf:>11.2f} {r["tokens_per_step"]:>11.1f} '
            f'{r["tokens_per_s"]:>9,.0f} {r["preemptions"]:>11}')
    v, o = BATCH[ds]['vLLM'], BATCH[ds]['Orca (Oracle)']
    log(f'{"":<9} vLLM against Orca (Oracle): {v["tokens_per_step"] / o["tokens_per_step"]:.1f}x the tokens per step, '
        f'{v["tokens_per_s"] / o["tokens_per_s"]:.1f}x the tokens per second (bigger batches make each step slower)')
    v = BATCH[ds]['vLLM']['sim']
    log(f'{"":<9} vLLM runs {v / BATCH[ds]["Orca (Oracle)"]["sim"]:.1f}x the requests of Orca (Oracle) and '
        f'{v / BATCH[ds]["Orca (Max)"]["sim"]:.1f}x those of Orca (Max)')
log('paper, Figure 13 (OPT-13B, A100): ShareGPT 7.00, 9.81, 13.62, 30.42; Alpaca 7.00, 43.24, 72.75, 132.44')
log(f'note: we cap a batch at 256 requests, as vLLM did by default')
OUT['batch'] = BATCH

# ---------------------------------------------------------------- 6. Figure 15: sharing the prompt in parallel sampling
log('')
log('== 6. Figure 15a rebuilt: memory saved by sharing prompt blocks in parallel sampling (block size 16)')


def sharing_refcount(p, o, n, block=16):
    """Method 1: a block manager. The prompt blocks are mapped by all n samples (reference count n). When a sample
    writes into a block that others still map, it gets its own copy first (copy-on-write). Each step we count the
    physical blocks in use and the blocks that n separate copies would need."""
    ref, nxt = {}, 0

    def new_block():
        nonlocal nxt
        ref[nxt] = 1; nxt += 1
        return nxt - 1
    prompt_blocks = [new_block() for _ in range(-(-p // block))]
    for b in prompt_blocks:
        ref[b] = n
    tables, length = [list(prompt_blocks) for _ in range(n)], [p] * n
    with_sharing = without = 0
    for _ in range(o):                                         # each step, every sample writes one token
        for s in range(n):
            if length[s] % block == 0:
                tables[s].append(new_block())                 # the last block is full: take a fresh one
            elif ref[tables[s][-1]] > 1:                       # shared and about to change: copy-on-write
                ref[tables[s][-1]] -= 1; tables[s][-1] = new_block()
            length[s] += 1
        with_sharing += sum(1 for c in ref.values() if c > 0)
        without += sum(len(t) for t in tables)
    return with_sharing, without


def sharing_closed(p, o, n, block=16):
    """Method 2: after the first step only the full prompt blocks, floor(p / block), stay shared."""
    full = p // block
    w = wo = 0
    for g in range(1, o + 1):
        own = -(-(p + g) // block)
        wo += n * own; w += full + n * (own - full)
    return w, wo


SHARE = {}
for ds, reqs in TRACE.items():
    SHARE[ds] = {}
    for n_out in (2, 4, 6):
        a = [sharing_refcount(p, o, n_out) for p, o in reqs[:300]]
        b = [sharing_closed(p, o, n_out) for p, o in reqs[:300]]
        save1 = 100 * (1 - sum(x for x, _ in a) / sum(y for _, y in a))
        save2 = 100 * (1 - sum(x for x, _ in b) / sum(y for _, y in b))
        SHARE[ds][n_out] = save1
        log(f'{ds:<9} {n_out} samples: saved {save1:5.2f}% (block manager), {save2:5.2f}% (closed form)')
log('paper, Figure 15a (Alpaca): 6.09%, 8.53%, 9.79%; §6.3 (ShareGPT): 16.2% to 30.5%')
log('assumption: all samples of a request write as many tokens as the dataset answer (real samples differ)')
OUT['sharing'] = SHARE

# ---------------------------------------------------------------- 7. Figure 18b: block size
log('')
log('== 7. Figure 18b rebuilt: vLLM in the simulator with block sizes 1 to 256 (memory only, no kernel cost)')
BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
BS_RATE = {ds: round(CAP[ds]['vLLM'], 1) for ds in TRACE}   # where vLLM with 16-token blocks reaches the line
BLOCKS = {}
for ds, reqs in TRACE.items():
    arrive = poisson(N_REQ, BS_RATE[ds], seed=2)
    BLOCKS[ds] = [simulate('vLLM', reqs, arrive, block=b)['norm_latency'] for b in BS]
    log(f'{ds} at {BS_RATE[ds]} req/s: ' + ', '.join(f'{b}: {v:.3f}' for b, v in zip(BS, BLOCKS[ds])))
OUT['blocks'] = {'sizes': BS, 'rate': BS_RATE, 'latency': BLOCKS}

# ---------------------------------------------------------------- 8. Figure 19a: swap against recompute
log('')
log('== 8. Figure 19a rebuilt: move a 1,024-token KV cache (OPT-125M, FP16) GPU -> CPU -> GPU, block by block,')
log('       against recomputing it with one prompt pass (laptop, fastest of 3 runs)')
CTX = 1024
ids = torch.randint(100, 20000, (1, CTX), device=DEV)
with torch.no_grad():
    cache = model(ids, use_cache=True).past_key_values
gpu = [t for layer in cache.layers for t in (layer.keys, layer.values)]       # 24 tensors of (1, 12, 1024, 64)
cpu = [torch.empty(t.shape, dtype=t.dtype) for t in gpu]
kv_mb = sum(t.numel() * t.element_size() for t in gpu) / 1e6
with torch.no_grad():
    t_re = fastest(lambda: model(ids, use_cache=True), 3)
log(f'KV cache: {kv_mb:.1f} MB in {len(gpu)} tensors; recompute (one prompt pass): {t_re * 1000:.1f} ms')
log(f'{"block":>5} {"copies":>7} {"swap out ms":>11} {"swap in ms":>10}')
SWAP = {}
for b in BS:
    def out():
        for g, c in zip(gpu, cpu):
            for j in range(0, CTX, b):
                c[:, :, j:j + b].copy_(g[:, :, j:j + b])

    def back():
        for g, c in zip(gpu, cpu):
            for j in range(0, CTX, b):
                g[:, :, j:j + b].copy_(c[:, :, j:j + b])
    t_out, t_in = fastest(out, 3), fastest(back, 3)
    SWAP[b] = {'out_ms': t_out * 1000, 'in_ms': t_in * 1000}
    log(f'{b:>5} {len(gpu) * CTX // b:>7,} {t_out * 1000:>11.1f} {t_in * 1000:>10.1f}')
OUT['swap'] = {'recompute_ms': t_re * 1000, 'kv_mb': kv_mb, 'blocks': SWAP}
best = min(BS, key=lambda b: SWAP[b]['out_ms'] + SWAP[b]['in_ms'])
faster = [b for b in BS if SWAP[b]['out_ms'] + SWAP[b]['in_ms'] < t_re * 1000]
log(f'swap in + out against recompute: {(SWAP[1]["out_ms"] + SWAP[1]["in_ms"]) / (t_re * 1000):.0f}x slower at block size 1, '
    f'{(SWAP[256]["out_ms"] + SWAP[256]["in_ms"]) / (t_re * 1000):.1f}x at block size 256; '
    + (f'faster from block size {faster[0]} up' if faster else 'never faster on this laptop'))

save('part5', OUT)
