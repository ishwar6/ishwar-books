"""Prefix-cache simulations for Part 5. Pure Python + NumPy, runs on a CPU. NOT SGLang or vLLM.

What this file models, and what it leaves out:
- RadixLRU: a token-level radix tree (page size 1 by default) with reference counts and
  leaf-first LRU eviction, as described in Section 3 of the SGLang paper.
- BlockLRU: a hash-chained full-block cache with an LRU free queue, freed in reverse block
  order, as described in vLLM's prefix-caching design document.
- Requests are prefilled one at a time. No batching, no decode, no real tensors, no GPU.
- Times come from a cost model fitted to real prefill timings (results/measure_prefix.json:
  Qwen2.5-0.5B, BF16, Apple M5 Pro). They are estimates, not engine benchmarks.
Run: python code/sglang/simulate.py  -> results/simulate.json and results/simulate_stdout.txt
"""
import hashlib, json, math, random
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = []
def say(s=''): OUT.append(s); print(s)


# ---------------------------------------------------------------- caches
class Node:
    __slots__ = ('edge', 'children', 'parent', 'last', 'ref')
    def __init__(self, edge=(), parent=None):
        self.edge, self.children, self.parent, self.last, self.ref = tuple(edge), {}, parent, 0, 0


class RadixLRU:
    """Radix tree over token IDs; a node's KV lives as long as the node. Evicts the least
    recently used unlocked leaf first; a parent becomes a candidate once its children are gone."""
    def __init__(self, capacity, page=1, trim=False):
        self.root, self.capacity, self.page, self.size, self.clock = Node(), capacity, page, 0, 0
        self.trim = trim                              # ablation: shorten a leaf from its tail instead of deleting it

    def _walk(self, tokens):
        node, used, path = self.root, 0, []
        while used < len(tokens) and tokens[used] in node.children:
            child = node.children[tokens[used]]
            n = 0
            while n < len(child.edge) and used + n < len(tokens) and child.edge[n] == tokens[used + n]:
                n += 1
            used += n; path.append(child)
            if n < len(child.edge):
                break
            node = child
        return used, path

    def match(self, tokens):
        used, path = self._walk(tuple(tokens))
        self.clock += 1
        for n in path:
            n.last = self.clock                       # a hit refreshes recency
        return used // self.page * self.page

    def insert(self, tokens):
        tokens, node, i = tuple(tokens), self.root, 0
        self.clock += 1
        locked = []
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                new = Node(tokens[i:], node); node.children[tokens[i]] = new
                self.size += len(new.edge); new.last = self.clock; locked.append(new)
                break
            n = 0
            while n < len(child.edge) and i + n < len(tokens) and child.edge[n] == tokens[i + n]:
                n += 1
            if n < len(child.edge):                   # split the edge where the sequences differ
                mid = Node(child.edge[:n], node); mid.last = child.last
                node.children[tokens[i]] = mid
                child.edge = child.edge[n:]; child.parent = mid
                mid.children[child.edge[0]] = child
                child = mid
            child.last = self.clock; locked.append(child)
            node, i = child, i + n
        for n in locked: n.ref += 1                   # the running request holds its path
        self.evict()
        for n in locked: n.ref -= 1

    def leaves(self):
        stack, out = [self.root], []
        while stack:
            n = stack.pop()
            if n is not self.root and not n.children and n.ref == 0:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def evict(self):
        while self.size > self.capacity:
            cands = self.leaves()
            if not cands:
                break
            victim = min(cands, key=lambda n: n.last)
            over = self.size - self.capacity
            if self.trim and over < len(victim.edge):
                victim.edge = victim.edge[:len(victim.edge) - over]; self.size -= over
                continue
            del victim.parent.children[victim.edge[0]]
            self.size -= len(victim.edge)

    def stored_tokens(self):
        return self.size


class BlockLRU:
    """Full-block prefix cache keyed by a hash chain (parent hash + this block's tokens).
    Only full blocks are cached. When a request ends, its blocks join the free queue in
    reverse order, so a request's last block is evicted before its first."""
    def __init__(self, capacity_tokens, block=16):
        self.B, self.cap = block, capacity_tokens // block
        self.order, self.clock = {}, 0                # hash -> eviction key; smallest goes first

    def keys(self, tokens):
        parent = b''
        for i in range(0, len(tokens) - self.B + 1, self.B):
            parent = hashlib.sha256(parent + repr(tuple(tokens[i:i + self.B])).encode()).digest()
            yield parent

    def match(self, tokens):
        n = 0
        for k in self.keys(tokens):
            if k not in self.order:
                break
            n += self.B
        return n

    def insert(self, tokens):
        self.clock += 1
        ks = list(self.keys(tokens))
        for idx, k in enumerate(ks):                  # freed in reverse: last block gets the oldest key
            self.order[k] = (self.clock, len(ks) - 1 - idx)
        while len(self.order) > self.cap:
            victim = min(self.order, key=self.order.get)
            del self.order[victim]


# ---------------------------------------------------------------- cost model from real timings
def fit_cost():
    m = json.loads((HERE / 'results/measure_prefix.json').read_text())
    rows = []
    for r in m['prefix_sweep'] + m['suffix_sweep']:
        P, S = r['P'], r['S']
        rows.append((P + S, 0, r['cold_ms']))         # cold: everything is new
        rows.append((S, P, r['warm_ms']))             # warm: S new tokens on top of P cached
    X = np.array([[1.0, n, n * (c + n / 2)] for n, c, _ in rows])
    y = np.array([t for *_, t in rows])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    err = float(np.max(np.abs(pred - y) / y))
    return coef, err


COEF, FIT_ERR = fit_cost()
def cost_ms(new, cached):
    a, b, c = COEF
    return a + b * new + c * new * (cached + new / 2)


# ---------------------------------------------------------------- 1. KV bytes per token
def kv_bytes():
    models = {  # layers, kv heads, head dim (bytes per value = 2, BF16)
        'Qwen2.5-0.5B': (24, 2, 64),
        'Qwen2.5-7B': (28, 4, 128),
        'Llama 3.1 8B': (32, 8, 128),
    }
    say('KV CACHE BYTES PER TOKEN  (2 x layers x kv_heads x head_dim x 2 bytes)')
    out = {}
    for name, (L, H, d) in models.items():
        b = 2 * L * H * d * 2
        handbook = b * 6000
        out[name] = {'layers': L, 'kv_heads': H, 'head_dim': d, 'bytes_per_token': b,
                     'handbook_6000_tokens_MiB': handbook / 2**20}
        say(f'  {name:13s} 2 x {L} x {H} x {d} x 2 = {b:7,d} bytes = {b/1024:5.0f} KiB'
            f'   6,000-token handbook: {handbook/2**20:6.1f} MiB')
    return out


# ---------------------------------------------------------------- 2. FLOPs saved by a cached prefix
def flops():
    L, d_attn, N = 24, 896, 357_898_112   # Qwen2.5-0.5B: layers, query width, non-embedding params
    def F(new, cached):                   # 2N per token for the weights, plus QK^T and AV over the context
        return 2 * N * new + 4 * L * d_attn * new * (cached + (new + 1) / 2)
    say(''); say('PREFILL FLOPs, Qwen2.5-0.5B  (2N per token + 4 L d per (query, key) pair)')
    rows = []
    m = json.loads((HERE / 'results/measure_prefix.json').read_text())
    for r in m['prefix_sweep']:
        P, S = r['P'], r['S']
        cold, warm = F(P + S, 0), F(S, P)
        rows.append({'P': P, 'S': S, 'cold_gflop': cold / 1e9, 'warm_gflop': warm / 1e9,
                     'flop_ratio': cold / warm, 'measured_ratio': r['speedup']})
        say(f'  P={P:5d} S={S}: cold {cold/1e9:7.1f} GFLOP, warm {warm/1e9:5.1f} GFLOP,'
            f' FLOP ratio {cold/warm:6.1f}x, measured time ratio {r["speedup"]:5.1f}x')
    # Scale check for a bigger model on a data-centre GPU: pure arithmetic, not a measurement.
    NL, LL, dL, peak = 6.98e9, 32, 4096, 989e12      # Llama 3.1 8B non-embedding params; H100 dense BF16 peak
    big = 2 * NL * 8192 + 4 * LL * dL * 8192 * (8192 + 1) / 2
    say(f'  Llama 3.1 8B, 8,192-token prefix: {big/1e12:.0f} TFLOP = {big/peak*1000:.0f} ms at 100% of an H100 (989 TFLOP/s)')
    return {'non_embedding_params': N, 'layers': L, 'query_width': d_attn, 'rows': rows,
            'llama8b_8192_tflop': big / 1e12, 'llama8b_8192_ideal_h100_ms': big / peak * 1000}


def end_to_end():
    m = json.loads((HERE / 'results/measure_prefix.json').read_text())
    r = next(x for x in m['prefix_sweep'] if x['P'] == 2048); tpot = m['decode_ms_at_2112']
    say(''); say(f'WHOLE REQUEST  P=2048, S=64, decode {tpot:.2f} ms per token (all measured)')
    out = []
    for n in [1, 20, 200, 1000]:
        cold = r['cold_ms'] + (n - 1) * tpot; warm = r['warm_ms'] + (n - 1) * tpot
        out.append({'answer_tokens': n, 'cold_ms': cold, 'warm_ms': warm, 'speedup': cold / warm})
        say(f'  answer {n:5d} tokens: cold {cold:8.1f} ms, warm {warm:8.1f} ms, speedup {cold/warm:5.2f}x')
    for h in [0.5, 0.9]:
        e = (1 - h) * r['cold_ms'] + h * r['warm_ms']
        say(f'  expected prefill at hit probability {h:.0%}: {e:.1f} ms')
    return out


# ---------------------------------------------------------------- 3. memory shared by the tree
def sharing():
    instr, book, q, sections, active = 512, 1536, 64, 4, 32
    seqs = [[0] * 0 + list(range(instr)) + [10_000 * (1 + i % sections) + k for k in range(book)]
            + [1_000_000 + i * 100 + k for k in range(q)] for i in range(active)]
    r = RadixLRU(10**9); [r.insert(s) for s in seqs]
    naive = sum(map(len, seqs)); tree = r.stored_tokens()
    say(''); say(f'MEMORY SHARING  {active} active requests: {instr} instruction + {book} handbook (1 of {sections}) + {q} question tokens')
    say(f'  no sharing : {naive:,} token slots   tree: {tree:,} token slots   ratio {naive/tree:.1f}x')
    for name, b in [('Qwen2.5-0.5B', 12288), ('Llama 3.1 8B', 131072)]:
        say(f'  {name:13s} {naive*b/2**30:5.2f} GiB  ->  {tree*b/2**30:5.2f} GiB')
    return {'active': active, 'naive_tokens': naive, 'tree_tokens': tree, 'ratio': naive / tree,
            'formula_tree': instr + sections * book + active * q}


# ---------------------------------------------------------------- 4. LRU eviction trace
def lru_trace():
    say(''); say('LRU EVICTION TRACE  capacity 10 tokens, leaf-first')
    r = RadixLRU(10)
    steps = [('handbook + q1', [1, 2, 3, 4, 5, 6, 20, 21]), ('handbook + q2', [1, 2, 3, 4, 5, 6, 30, 31]),
             ('handbook + q3', [1, 2, 3, 4, 5, 6, 40, 41]), ('handbook + q2 again', [1, 2, 3, 4, 5, 6, 30, 31]),
             ('handbook + q1 again', [1, 2, 3, 4, 5, 6, 20, 21])]
    trace = []
    for name, s in steps:
        hit = r.match(s); r.insert(s)
        def walk(n, pre):
            out = []
            for c in n.children.values():
                out.append(pre + ' '.join(map(str, c.edge)))
                out += walk(c, pre + '  ')
            return out
        tree = walk(r.root, '')
        say(f'  {name:20s} reuse {hit}, compute {len(s)-hit}; stored {r.size} tokens: ' + ' | '.join(t.strip() for t in tree))
        trace.append({'request': name, 'reused': hit, 'computed': len(s) - hit, 'stored': r.size, 'edges': [t.strip() for t in tree]})
    return trace


# ---------------------------------------------------------------- 5. block alignment loss
def alignment():
    rng = random.Random(3)
    lens = [rng.randrange(0, 4096) for _ in range(100_000)]
    say(''); say('BLOCK ALIGNMENT  shared prefix length uniform in 0..4095, tokens lost to rounding down')
    out = {}
    for B in [1, 16, 32, 64, 128]:
        lost = sum(L - L // B * B for L in lens) / len(lens)
        out[B] = {'mean_lost': lost, 'formula': (B - 1) / 2}
        say(f'  block {B:3d}: mean lost {lost:6.2f} tokens   formula (B-1)/2 = {(B-1)/2:6.1f}')
    return out


# ---------------------------------------------------------------- 6. capacity sweep, realistic mix
def workload(seed=7, n=600):
    rng = random.Random(seed)
    instr = list(range(1, 301))                                   # 300-token shared instructions
    books = [[100_000 * (b + 1) + k for k in range(rng.randrange(900, 2100))] for b in range(6)]
    weights = [1 / (b + 1) for b in range(6)]                     # Zipf-like handbook popularity
    reqs, sessions, uid = [], [], 10**7
    for _ in range(n):
        if sessions and rng.random() < 0.35:                      # a follow-up in an existing chat
            s = rng.choice(sessions)
            q = [uid + k for k in range(rng.randrange(20, 90))]; uid += 1000
            s.extend(q); prompt = list(s)
        else:
            b = rng.choices(range(6), weights)[0]
            q = [uid + k for k in range(rng.randrange(20, 90))]; uid += 1000
            prompt = instr + books[b] + q
            s = list(prompt); sessions.append(s)
            if len(sessions) > 40: sessions.pop(0)
        ans = [uid + k for k in range(rng.randrange(40, 160))]; uid += 1000
        reqs.append((prompt, ans))
        if s is not None and prompt is not None and prompt[-len(q):] == q and s[-len(q):] == q:
            s.extend(ans)                                         # the answer becomes chat history
    return reqs


def run_cache(cache, reqs):
    total = hit = 0
    for prompt, ans in reqs:
        m = cache.match(prompt)
        m = min(m, len(prompt) - 1)                               # the last prompt token is always computed
        hit += m; total += len(prompt)
        cache.insert(prompt + ans)                                # generated tokens are cached too
    return hit / total, total - hit


def capacity_sweep():
    reqs = workload()
    say(''); say(f'CAPACITY SWEEP  {len(reqs)} requests: 6 handbooks (900-2,100 tokens), Zipf popularity, 35% follow-ups')
    say('  capacity   radix(page 1)   radix(trim tails)   blocks(16)   blocks(64)')
    rows = []
    for cap in [4096, 8192, 16384, 32768, 65536]:
        h1, _ = run_cache(RadixLRU(cap), reqs)
        h2, _ = run_cache(BlockLRU(cap, 16), reqs)
        h3, _ = run_cache(BlockLRU(cap, 64), reqs)
        h4, _ = run_cache(RadixLRU(cap, trim=True), reqs)
        rows.append({'capacity_tokens': cap, 'radix_page1': h1, 'radix_trim': h4, 'block16': h2, 'block64': h3})
        say(f'  {cap:8,d}   {h1:13.1%}   {h4:17.1%}   {h2:10.1%}   {h3:10.1%}')
    return rows


# ---------------------------------------------------------------- 7. scheduling order, offline batch
INSTR, BOOK, Q = 512, 1536, 64
def hb_request(book, i):
    return list(range(1, INSTR + 1)) + [10_000 * (book + 1) + k for k in range(BOOK)] + [5_000_000 + 1000 * i + k for k in range(Q)]


def schedule(order_name, reqs, capacity, arrivals=None, seed=0):
    """One server, one prefill at a time. Each step picks a waiting request by policy."""
    cache, rng = RadixLRU(capacity), random.Random(seed)
    arrivals = arrivals or [0.0] * len(reqs)
    waiting, t, done, computed, total, events = [], 0.0, {}, 0, 0, []
    pending = sorted(range(len(reqs)), key=lambda i: (arrivals[i], i))
    while pending or waiting:
        while pending and arrivals[pending[0]] <= t:
            waiting.append(pending.pop(0))
        if not waiting:
            t = arrivals[pending[0]]; continue
        if order_name == 'fcfs':
            pick = waiting[0]
        elif order_name == 'random':
            pick = rng.choice(waiting)
        else:                                         # lpm: longest cached prefix first, ties by arrival
            pick = max(waiting, key=lambda i: (cache._walk(tuple(reqs[i]))[0], -i))
        waiting.remove(pick)
        p = reqs[pick]
        m = min(cache.match(p), len(p) - 1)
        start = t
        t += cost_ms(len(p) - m, m)
        events.append({'req': pick, 'book': p[INSTR] // 10_000 - 1, 'start': round(start, 2), 'end': round(t, 2), 'computed': len(p) - m})
        cache.insert(p)
        done[pick] = t - arrivals[pick]
        computed += len(p) - m; total += len(p)
    ttft = list(done.values())
    return {'computed_tokens': computed, 'hit_rate': 1 - computed / total, 'total_ms': t,
            'mean_ttft_ms': sum(ttft) / len(ttft), 'max_ttft_ms': max(ttft), 'ttft': done, 'timeline': events}


def offline():
    reqs = [hb_request(i % 4, i) for i in range(32)]              # arrival order A B C D A B C D ...
    cap = INSTR + 2 * BOOK + 8 * Q                                 # room for two handbooks
    optimal = INSTR + 4 * BOOK + 32 * Q
    say(''); say(f'SCHEDULING ORDER  32 waiting requests (A B C D repeated), cache {cap:,} tokens = 2 handbooks')
    say(f'  each: {INSTR} instructions + {BOOK} handbook + {Q} question; best possible = {optimal:,} tokens computed')
    say('  policy   tokens computed   hit rate   all done   mean wait   worst wait')
    out = {'capacity': cap, 'optimal_computed': optimal}
    for pol in ['fcfs', 'random', 'lpm']:
        r = schedule(pol, reqs, cap, seed=1)
        out[pol] = {k: v for k, v in r.items() if k != 'ttft' and (k != 'timeline' or pol != 'random')}
        say(f'  {pol:6s}   {r["computed_tokens"]:15,d}   {r["hit_rate"]:8.1%}   {r["total_ms"]:6.0f} ms'
            f'   {r["mean_ttft_ms"]:6.0f} ms   {r["max_ttft_ms"]:7.0f} ms')
    return out


def starvation():
    """Online: a busy stream of handbook-A questions, and one handbook-D question that arrives early."""
    reqs, arr = [], []
    for i in range(60):
        reqs.append(hb_request(0, i)); arr.append(10.0 * i)      # A every 10 ms, each ~15 ms warm: queue grows
    reqs.append(hb_request(3, 999)); arr.append(25.0)             # one D request at 25 ms
    cap = 16_000
    say(''); say('STARVATION  60 handbook-A questions every 10 ms + one handbook-D question at 25 ms')
    out = {}
    for pol in ['fcfs', 'lpm']:
        r = schedule(pol, reqs, cap, arrivals=arr)
        d = r['ttft'][60]
        a = [v for k, v in r['ttft'].items() if k != 60]
        out[pol] = {'d_wait_ms': d, 'a_mean_ms': sum(a) / len(a), 'a_max_ms': max(a), 'total_ms': r['total_ms'],
                    'd_start_ms': next(e['start'] for e in r['timeline'] if e['req'] == 60)}
        say(f'  {pol:4s}: D waits {d:6.0f} ms   A mean {sum(a)/len(a):5.0f} ms, A worst {max(a):5.0f} ms')
    return out


# ---------------------------------------------------------------- 8. jump-forward step count
def jump_forward():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')
    # The model decides where a free string ends, so the closing quote of "Asha Rao" is free;
    # after the enum value "annual" the closing quote is forced. Integers end when the model stops.
    parts = [('forced', '{"employee": "'), ('free', 'Asha Rao"'), ('forced', ', "leave_type": "'),
             ('free', 'annual'), ('forced', '", "days": '), ('free', '12'), ('forced', '}')]
    full = ''.join(p for _, p in parts)
    enc = tok(full, return_offsets_mapping=True, add_special_tokens=False)
    bounds, pos = [], 0
    for kind, p in parts:
        bounds.append((pos, pos + len(p), kind)); pos += len(p)
    kinds = []
    for a, b in enc['offset_mapping']:
        k = next(kind for s, e, kind in bounds if s <= a < e)
        kinds.append(k)
    # a token that starts in a forced segment is forced; consecutive forced tokens form one jump
    normal = len(kinds)
    free = kinds.count('free')
    jumps = sum(1 for i, k in enumerate(kinds) if k == 'forced' and (i == 0 or kinds[i - 1] != 'forced'))
    say(''); say('JUMP-FORWARD  Qwen2.5 tokenizer, one JSON answer')
    say(f'  {full}')
    say(f'  tokens {normal}: forced {normal-free}, free {free}; forced runs {jumps}')
    say(f'  forward passes: token by token {normal}, with jump-forward {free + jumps}')
    return {'text': full, 'tokens': normal, 'free': free, 'forced': normal - free, 'forced_runs': jumps,
            'passes_normal': normal, 'passes_jump': free + jumps,
            'tokens_text': [tok.decode([t]) for t in enc['input_ids']], 'kinds': kinds}


def main():
    say('$ python code/sglang/simulate.py')
    say(f'cost model fitted to measured M5 Pro prefill: t(n, c) = {COEF[0]:.1f} + {COEF[1]:.4f} n + {COEF[2]*1e6:.2f}e-6 n (c + n/2) ms,'
        f' worst error {FIT_ERR:.0%}')
    res = {'kind': 'CPU simulations; times from a cost model fitted to real Qwen2.5-0.5B timings on an Apple M5 Pro; not engine benchmarks',
           'cost_model': {'a_ms': COEF[0], 'b_ms_per_token': COEF[1], 'c_ms_per_pair': COEF[2], 'max_rel_error': FIT_ERR}}
    res['kv_bytes'] = kv_bytes()
    res['flops'] = flops()
    res['end_to_end'] = end_to_end()
    res['sharing'] = sharing()
    res['lru_trace'] = lru_trace()
    res['alignment'] = alignment()
    res['capacity_sweep'] = capacity_sweep()
    res['offline_scheduling'] = offline()
    res['starvation'] = starvation()
    res['jump_forward'] = jump_forward()
    (HERE / 'results/simulate.json').write_text(json.dumps(res, indent=2) + '\n')
    (HERE / 'results/simulate_stdout.txt').write_text('\n'.join(OUT) + '\n')


if __name__ == '__main__':
    main()
