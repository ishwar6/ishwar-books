"""Background page of vLLM, explained: the KV cache and paging, run on a laptop.

What it shows (every number is printed and saved to results/partb_stdout.txt and results/partb.json):
  1. Attention on three tiny tokens: by hand (Method 1) and with PyTorch's scaled_dot_product_attention (Method 2).
  2. Why keys and values can be cached: in OPT-125M, the keys of the first 8 tokens do not change when 9 more
     tokens follow them (the causal mask).
  3. One new token needs only its own query plus the cached keys and values: layer 0 of OPT-125M rebuilt by hand
     from the cache (Method 1) against the library's own attention output (Method 2).
  4. Contiguous memory: a first-fit allocator that refuses a request although enough memory is free (external
     fragmentation), against fixed-size blocks; and the buddy allocator's rounding to powers of two.
  5. Paging: a virtual address translated by division (Method 1) and by bit shifts as hardware does (Method 2);
     this laptop's real page size; and pages given on demand by the real operating system (mmap and RSS).
Model: facebook/opt-125m in FP32 on the CPU (exact comparisons). Run: python vllm_partb.py   (under a minute)"""
import math, mmap, os, random
import psutil
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save

log = Log('partb')
torch.manual_seed(0)
OUT = {}

# ---------------------------------------------------------------- 1. attention on three tiny tokens
log('== 1. Attention for one query over three tokens (d = 2)')
q = torch.tensor([1., 0.])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])        # one key per earlier token
V = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])        # one value per earlier token


def attention_by_hand(q, K, V):
    """softmax(q K^T / sqrt(d)) V, one step at a time."""
    d = q.shape[-1]
    scores = K @ q / math.sqrt(d)                         # how well the query matches each key
    weights = torch.exp(scores) / torch.exp(scores).sum() # softmax: positive, sums to 1
    return scores, weights, weights @ V                   # weighted average of the values


scores, weights, out1 = attention_by_hand(q, K, V)
out2 = F.scaled_dot_product_attention(q[None, None, None], K[None, None], V[None, None])[0, 0, 0]
log('scores  q.k/sqrt(2):', [round(v, 3) for v in scores.tolist()])
log('weights softmax    :', [round(v, 3) for v in weights.tolist()], f'sum {weights.sum():.3f}')
log('output, Method 1   :', [round(v, 3) for v in out1.tolist()])
log('output, Method 2   :', [round(v, 3) for v in out2.tolist()], f'agree: {torch.allclose(out1, out2, atol=1e-6)}')
OUT['tiny'] = {'scores': scores.tolist(), 'weights': weights.tolist(), 'out': out1.tolist()}

# ---------------------------------------------------------------- 2. earlier keys never change
log('')
log('== 2. Do the keys of earlier tokens change when more tokens follow? (OPT-125M, FP32, CPU)')
tok = AutoTokenizer.from_pretrained('facebook/opt-125m')
model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', dtype=torch.float32).eval()
ids = tok('Four score and seven years ago our fathers brought forth on this continent a new nation',
          return_tensors='pt').input_ids
N_SHORT = 8
with torch.no_grad():
    short = model(ids[:, :N_SHORT], use_cache=True).past_key_values
    full = model(ids, use_cache=True).past_key_values
diffs = [max((short.layers[l].keys - full.layers[l].keys[:, :, :N_SHORT]).abs().max().item(),
             (short.layers[l].values - full.layers[l].values[:, :, :N_SHORT]).abs().max().item())
         for l in range(len(short.layers))]
log(f'{N_SHORT} tokens alone, then the same {N_SHORT} tokens followed by {ids.shape[1] - N_SHORT} more')
log(f'largest change of a key or value of the first {N_SHORT} tokens, over all 12 layers: {max(diffs):.2e}')
log(f'largest key value, for scale: {full.layers[11].keys.abs().max().item():.1f}')
OUT['unchanged_max_diff'] = max(diffs)

# ---------------------------------------------------------------- 3. one new token from the cache
log('')
log('== 3. Layer 0 for the newest token, rebuilt from the cache by hand')
attn = model.model.decoder.layers[0].self_attn
seen = {}


def grab_in(mod, args, kwargs):
    seen['h'] = kwargs.get('hidden_states', args[0] if args else None).detach().clone()


def grab_out(mod, args, out):
    seen['out'] = out[0].detach().clone()


h1 = attn.register_forward_pre_hook(grab_in, with_kwargs=True)
h2 = attn.register_forward_hook(grab_out)
with torch.no_grad():
    cache = model(ids[:, :-1], use_cache=True).past_key_values     # prompt phase: every token but the last
    model(ids[:, -1:], past_key_values=cache, use_cache=True)       # generation phase: the last token alone
h1.remove(); h2.remove()


def layer0_from_cache(h, cache):
    """Method 1: query of the new token only; keys and values of all tokens read from the cache."""
    H, D = attn.num_heads, attn.head_dim
    qn = (attn.q_proj(h) * attn.scaling).view(1, 1, H, D).transpose(1, 2)   # (1, 12, 1, 64); OPT scales q
    Kc, Vc = cache.layers[0].keys, cache.layers[0].values                   # (1, 12, n, 64), new token included
    w = torch.softmax(qn @ Kc.transpose(-1, -2), dim=-1)                    # (1, 12, 1, n): one weight per token
    o = (w @ Vc).transpose(1, 2).reshape(1, 1, H * D)                       # weighted values, heads joined
    return attn.out_proj(o), w


with torch.no_grad():
    ours, w = layer0_from_cache(seen['h'], cache)
n_all = cache.layers[0].keys.shape[2]
log(f'input to layer 0: {tuple(seen["h"].shape)} (one token); cached keys: {tuple(cache.layers[0].keys.shape)}')
log(f'attention weights of head 0 over the {n_all} tokens: ' + ' '.join(f'{v:.2f}' for v in w[0, 0, 0].tolist()))
diff = (ours - seen['out']).abs().max().item()
log(f'Method 1 (by hand from the cache) vs Method 2 (the library): max difference {diff:.2e}')
OUT['layer0_diff'] = diff

# ---------------------------------------------------------------- 4. contiguous memory and fragmentation
log('')
log('== 4. Contiguous memory: a first-fit allocator (memory of 32 slots)')


class FirstFit:
    """Give each request one contiguous run of slots: the first free gap that is large enough."""
    def __init__(self, size):
        self.size, self.used = size, {}                  # name -> (start, length)

    def gaps(self):
        out, pos = [], 0
        for s, n in sorted(self.used.values()):
            if s > pos:
                out.append((pos, s - pos))
            pos = s + n
        if pos < self.size:
            out.append((pos, self.size - pos))
        return out

    def alloc(self, name, n):
        for s, g in self.gaps():
            if g >= n:
                self.used[name] = (s, n)
                return True
        return False

    def free(self, name):
        del self.used[name]

    def picture(self):
        row = ['.'] * self.size
        for name, (s, n) in self.used.items():
            row[s:s + n] = name * n
        return ''.join(row)


m = FirstFit(32)
for name, n in [('A', 8), ('B', 6), ('C', 8), ('D', 6)]:
    m.alloc(name, n)
log(f'after A=8, B=6, C=8, D=6:  {m.picture()}')
m.free('B'); m.free('D')
free_total = sum(g for _, g in m.gaps())
log(f'after B and D finish:      {m.picture()}   free {free_total}, largest gap {max(g for _, g in m.gaps())}')
ok = m.alloc('E', 12)
log(f'request E needs 12 slots in one piece: {"granted" if ok else "refused"}, although {free_total} slots are free')

log('the same memory in blocks of 2 slots, any block anywhere:')
blocks_free = free_total // 2
log(f'  free blocks {blocks_free}; E needs {math.ceil(12 / 2)} blocks: {"granted" if blocks_free >= 6 else "refused"}')

# a longer trace: requests come and go; a request that does not fit is refused (it would have to wait)
SIZE, BLOCK, N_REQ = 4096, 16, 5000
rng = random.Random(0)
trace = [(rng.randint(20, 400), rng.random(), rng.random()) for _ in range(N_REQ)]   # size, finish?, which one


def replay(contiguous):
    """Replay the same trace. Before each new request, one live request may finish (same random draws for both)."""
    mm, free, need, live = FirstFit(SIZE), SIZE // BLOCK, {}, []
    refused = refused_but_room = 0
    held, free_at_refusal, size_of = [], [], {}
    for i, (n, fin, pick) in enumerate(trace):
        if len(live) > 8 and fin < 0.5:
            j = live.pop(int(pick * len(live)))
            if contiguous:
                mm.free(j)
            else:
                free += need.pop(j)
        if contiguous:
            ok = mm.alloc(i, n)
            room = sum(g for _, g in mm.gaps()) >= n
        else:
            b = math.ceil(n / BLOCK)
            ok = b <= free
            if ok:
                free -= b; need[i] = b
            room = ok
        if ok:
            live.append(i); size_of[i] = n
        else:
            refused += 1
            refused_but_room += room                      # enough free memory in total, but not in one piece
            free_at_refusal.append(SIZE - sum(size_of[j] for j in live))
        held.append(sum(size_of[j] for j in live) / SIZE)  # share of memory that holds data of live requests
    return refused, refused_but_room, sum(held) / len(held), sum(free_at_refusal) / max(1, len(free_at_refusal))


rc, rc_room, uc, fc = replay(True)
rb, _, ub, fb = replay(False)
log(f'trace of {N_REQ} requests in {SIZE} slots, sizes 20 to 400; a request that does not fit is refused:')
log(f'  contiguous first fit: memory holding data {100 * uc:.1f}% on average; {rc} refused, '
    f'{rc_room} of them while enough slots were free in total; slots not holding data at a refusal: {fc:.0f} on average')
log(f'  blocks of {BLOCK}:         memory holding data {100 * ub:.1f}% on average; {rb} refused; '
    f'slots not holding data at a refusal: {fb:.0f} on average')
OUT['frag'] = {'contiguous': {'refused': rc, 'refused_with_room': rc_room, 'held': uc, 'free_at_refusal': fc},
               'blocks': {'refused': rb, 'held': ub, 'free_at_refusal': fb}}

log('')
log('== 4b. The buddy allocator rounds every request up to a power of two')
for n in [25, 300, 513, 1500]:
    r = 2 ** math.ceil(math.log2(n))
    log(f'  request {n:>5} -> chunk {r:>5}: {r - n:>4} slots never used ({100 * (r - n) / r:.0f}% of the chunk)')

# ---------------------------------------------------------------- 5. paging
log('')
log('== 5. Paging: from a virtual address to a physical one')
P = 4096
page_table = {0: 5, 1: 2, 2: 7, 3: 0}                    # virtual page -> physical page (frame)


def translate_div(v):
    """Method 1: page number and offset by division."""
    page, offset = v // P, v % P
    return page, offset, page_table[page] * P + offset


def translate_bits(v):
    """Method 2: what the hardware does. 4096 = 2^12, so shift by 12 bits and keep the low 12 bits."""
    page, offset = v >> 12, v & 0xFFF
    return page, offset, (page_table[page] << 12) | offset


for v in [10000, 4095, 4096]:
    a, b = translate_div(v), translate_bits(v)
    log(f'virtual {v:>6} -> page {a[0]}, offset {a[1]:>4} -> frame {page_table[a[0]]} -> physical {a[2]:>6}; '
        f'bit method agrees: {a == b}')
real_page = os.sysconf('SC_PAGE_SIZE')
log(f"this laptop's page size: {real_page} bytes ({real_page // 1024} KB)")
OUT['page_size'] = real_page

log('')
log('== 5b. Pages on demand: map 1 GB, then touch 256 MB of it')
proc = psutil.Process()
rss = lambda: proc.memory_info().rss
r0 = rss()
buf = mmap.mmap(-1, 1 << 30)                              # ask the OS for 1 GB of virtual memory
r1 = rss()
TOUCH = 256 << 20
for off in range(0, TOUCH, real_page):                    # write one byte in each page
    buf[off] = 1
r2 = rss()
log(f'resident memory before: {r0 / 2**20:8.1f} MB')
log(f'after mapping 1 GB:     {r1 / 2**20:8.1f} MB   (+{(r1 - r0) / 2**20:.1f} MB)')
log(f'after touching 256 MB:  {r2 / 2**20:8.1f} MB   (+{(r2 - r1) / 2**20:.1f} MB, {TOUCH // real_page} pages written)')
buf.close()
OUT['mmap'] = {'map_mb': (r1 - r0) / 2**20, 'touch_mb': (r2 - r1) / 2**20, 'pages': TOUCH // real_page}

log('')
log('== 5c. The waste of paging is the unused end of the last page')
for n in [1, 4096, 4097, 10000]:
    pages = math.ceil(n / P)
    log(f'  {n:>6} bytes -> {pages} pages of 4096 = {pages * P:>6} bytes, {pages * P - n:>4} unused')
save('partb', OUT)
