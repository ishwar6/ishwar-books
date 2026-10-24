"""Part 3 of vLLM, explained: PagedAttention, the block table and the KV cache manager (§4.1 to §4.3, Figures 4 to 7).

What it shows (every number is printed and saved to results/part3_stdout.txt and results/part3.json):
  1. Figure 6, step by step: the prompt "Four score and seven years ago our" (7 tokens) goes through a real
     OPT-125M, and its keys and values go into physical KV blocks of 4 tokens. We print the block table after the
     prefill and after each decoding step, as in the figure (logical blocks 0, 1, 2 -> physical blocks 7, 1, 3).
  2. Eq. 4 on real numbers: the query of "forth" against the three KV blocks of Figure 5. Method 1 is our own
     block-by-block attention (two passes: scores per block, then values per block). We print the share of
     attention that each block gets.
  3. Method 2, the library: the same keys and values from the model's own past_key_values, in one contiguous
     tensor, through torch scaled_dot_product_attention. Max difference per layer, and of the final logits.
  4. Block sizes 1, 4 and 16, with free blocks handed out in a random order: greedy generation of 48 tokens
     through our paged engine against model.generate. Same tokens, max difference of the logits.
  5. Figure 7: two requests in one pool of 9 physical blocks, their blocks interleaved. Each request's logits
     against the library run alone. Then request A ends, its blocks go back to the free list, a third request
     takes them.
  6. Waste per request on real Alpaca lengths (OPT tokenizer), for block sizes 1 to 128: Method 1 appends
     tokens one by one through a block table; Method 2 is the closed form B * ceil(L / B) - L. The waste is never
     more than B - 1 slots. Compared with a contiguous chunk of 2,048 slots per request.
  7. The cost side of the block size: our block-by-block attention over a 2,048-token cache, timed for block
     sizes 1 to 256 (fastest of 7 runs; the laptop was shared, so timings are noisy).
  8. The lookup: a token position splits into a logical block and an offset, exactly as a byte address splits
     into a page number and an offset. The keys found this way equal the library's contiguous cache.
  9. Why splitting the softmax is exact: the Background example by hand, block by block; then every block of the
     real "forth" query finished alone and merged at the end, against SDPA.
Model: facebook/opt-125m, FP32 on the CPU, so the checks are exact to rounding.
Run: python vllm_part3.py   (about a minute on an M-series Mac)"""
import math, random, time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save

log = Log('part3')
NAME = 'facebook/opt-125m'
torch.manual_seed(0)
OUT = {}

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
dec = model.model.decoder
LAYERS, H = model.config.num_hidden_layers, model.config.num_attention_heads
DH = model.config.hidden_size // H                            # head size d = 64 for OPT-125M


# ---------------------------------------------------------------- the KV cache manager (§4.2), from scratch
class BlockAllocator:
    """Hands out physical block numbers from a free list, and takes them back."""
    def __init__(self, order):
        self.free = list(order)

    def allocate(self):
        if not self.free:
            raise MemoryError('no free KV block')
        return self.free.pop(0)

    def release(self, block):
        self.free.append(block)


class PagedKVCache:
    """Physical KV blocks for every layer, plus one block table per request.
    K and V have shape (layers, physical blocks, block size B, heads, head size).
    A block table is a list of [physical block number, # filled], one entry per logical block."""
    def __init__(self, num_blocks, B, order=None):
        self.B = B
        self.K = torch.zeros(LAYERS, num_blocks, B, H, DH)
        self.V = torch.zeros(LAYERS, num_blocks, B, H, DH)
        self.alloc = BlockAllocator(order if order is not None else range(num_blocks))
        self.tables = {}

    def append_slot(self, req):
        """Reserve the slot for one new token: the next free position of the last logical block,
        or a new physical block when the last one is full. Returns (physical block, offset)."""
        table = self.tables.setdefault(req, [])
        if not table or table[-1][1] == self.B:                 # last block full (or none yet): map a new one
            table.append([self.alloc.allocate(), 0])
        phys, filled = table[-1]
        table[-1][1] += 1                                       # the "# filled" column of the block table
        return phys, filled

    def free(self, req):
        for phys, _ in self.tables.pop(req):
            self.alloc.release(phys)


# ---------------------------------------------------------------- PagedAttention, Eq. 4 (Method 1)
def paged_attention(q, cache, layer, table):
    """Eq. 4 for one query token. q: (H, d). table: [[physical block, # filled], ...] in logical order.
    Pass 1: the scores of each KV block. Pass 2: each block's values, weighted by its share of the softmax."""
    scale = 1 / math.sqrt(DH)
    scores = []
    for phys, filled in table:                                  # logical block j lives in physical block `phys`
        K_j = cache.K[layer, phys, :filled]                     # (filled, H, d): fetched from wherever it sits
        scores.append(torch.einsum('hd,thd->ht', q, K_j) * scale)          # q_i^T K_j / sqrt(d), (H, filled)
    m = torch.stack([s.max(-1).values for s in scores]).max(0).values      # subtract the max: same result, no overflow
    exps = [torch.exp(s - m[:, None]) for s in scores]
    denom = sum(e.sum(-1) for e in exps)                        # the sum over all ceil(i / B) blocks, (H,)
    o = torch.zeros(H, DH)
    for (phys, filled), e in zip(table, exps):
        A_ij = e / denom[:, None]                               # attention on block j, (H, filled)
        o += torch.einsum('ht,thd->hd', A_ij, cache.V[layer, phys, :filled])   # V_j A_ij^T
    return o


def causal_attention(q, k, v):
    """Conventional attention for the prefill step (§4.3): all prompt tokens at once, with a causal mask.
    q, k, v: (n, H, d) -> (n, H, d)."""
    n = q.shape[0]
    s = torch.einsum('ihd,jhd->hij', q, k) / math.sqrt(DH)
    s = s.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), float('-inf'))
    return torch.einsum('hij,jhd->ihd', torch.softmax(s, -1), v)


TRACE = {}                                                      # per-layer queries and outputs, for the checks


@torch.no_grad()
def forward(ids, start, req, cache, trace=False):
    """Run OPT-125M on new tokens `ids` (list) of request `req`, whose earlier tokens are already in `cache`.
    Several tokens = the prefill step (conventional attention); one token = a decoding step (PagedAttention).
    Each new token's key and value are written into the slot that the block table gives it."""
    n = len(ids)
    slots = [cache.append_slot(req) for _ in range(n)]          # the KV cache manager reserves the slots first
    pos = torch.arange(start, start + n)
    x = dec.embed_tokens(torch.tensor(ids)) + dec.embed_positions.weight[pos + 2]   # OPT's position offset is 2
    for l, layer in enumerate(dec.layers):
        a, h = layer.self_attn, layer.self_attn_layer_norm(x)
        q, k, v = (p(h).view(n, H, DH) for p in (a.q_proj, a.k_proj, a.v_proj))
        for t, (phys, off) in enumerate(slots):                 # write the new keys and values into their blocks
            cache.K[l, phys, off], cache.V[l, phys, off] = k[t], v[t]
        if n > 1:
            o = causal_attention(q, k, v)
        else:
            o = paged_attention(q[0], cache, l, cache.tables[req])[None]
            if trace:
                TRACE[l] = (q[0].clone(), o[0].clone())
        x = x + a.out_proj(o.reshape(n, -1))
        x = x + layer.fc2(layer.activation_fn(layer.fc1(layer.final_layer_norm(x))))
    return model.lm_head(dec.final_layer_norm(x))[-1]          # logits for the next token


def table_str(cache, req):
    return '  '.join(f'logical {j} -> physical {p} ({f} filled)' for j, (p, f) in enumerate(cache.tables[req]))


def words_in(cache, owner, words):
    """Which word sits in each slot of each physical block: {physical block: [word or '.']}."""
    pool = {}
    for req, table in cache.tables.items():
        w = words[req]
        for j, (phys, filled) in enumerate(table):
            pool[phys] = [w[j * cache.B + s] if s < filled else '.' for s in range(cache.B)]
            owner[phys] = req
    return pool


# ---------------------------------------------------------------- 1. Figure 6, step by step
log('== 1. Figure 6: the block table of request A, step by step (block size 4, real OPT-125M)')
PROMPT_A = tok('Four score and seven years ago our', add_special_tokens=False).input_ids
OUT_A = tok(' fathers brought forth', add_special_tokens=False).input_ids
WORDS = {'A': [tok.decode(t).strip() for t in PROMPT_A + OUT_A]}
log(f'prompt: {len(PROMPT_A)} tokens {WORDS["A"][:7]}; we then feed the outputs "fathers", "brought", "forth"')
ORDER = [7, 1, 5, 2, 3, 0, 4, 6, 8]                           # the free list, in the order the figures use
cacheA = PagedKVCache(9, 4, order=[7, 1, 3, 0, 2, 4, 5, 6, 8])
steps = []
logits = forward(PROMPT_A, 0, 'A', cacheA)
log(f'prefill  (7 tokens): {table_str(cacheA, "A")}')
log(f'          model\'s top next token: {tok.decode(logits.argmax())!r}')
steps.append({'step': 'prefill', 'table': [list(e) for e in cacheA.tables['A']]})
seen = list(PROMPT_A)
for i, t in enumerate(OUT_A):
    logits = forward([t], len(seen), 'A', cacheA, trace=(i == 2))
    seen.append(t)
    log(f'decode {i + 1} ({tok.decode(t).strip()!r:>9}): {table_str(cacheA, "A")}')
    steps.append({'step': f'decode {i + 1}', 'word': tok.decode(t).strip(),
                  'table': [list(e) for e in cacheA.tables['A']]})
owner = {}
pool = words_in(cacheA, owner, WORDS)
log('physical blocks now (9 blocks of 4 slots; "." = free slot, "-" = free block):')
for b in range(9):
    log(f'  block {b}: ' + (' '.join(f'{w:>8}' for w in pool[b]) if b in pool else '       -'))
OUT['fig6'] = {'steps': steps, 'pool': pool}

# ---------------------------------------------------------------- 2. Eq. 4 on real numbers
log('')
log('== 2. Eq. 4: the query of "forth" against KV blocks 0, 1 and 2 (layer 6, mean over the 12 heads)')
L6 = 6
q6, _ = TRACE[L6]
scale = 1 / math.sqrt(DH)
blocks = []
for j, (phys, filled) in enumerate(cacheA.tables['A']):
    blocks.append(torch.einsum('hd,thd->ht', q6, cacheA.K[L6, phys, :filled]) * scale)
allsc = torch.cat(blocks, -1)
A = torch.softmax(allsc, -1)                                    # (H, 10): one softmax over all blocks together
EQ4 = []
c = 0
for j, (phys, filled) in enumerate(cacheA.tables['A']):
    share = A[:, c:c + filled].mean(0)
    ws = WORDS['A'][j * 4: j * 4 + filled]
    log(f'block {j} (physical {phys}): ' + ', '.join(f'{w} {s:.3f}' for w, s in zip(ws, share.tolist()))
        + f'   block total {share.sum():.3f}')
    EQ4.append({'block': j, 'phys': phys, 'words': ws, 'weights': share.tolist()})
    c += filled
log(f'all ten weights add up to {A.mean(0).sum():.6f}')
lone = torch.softmax(blocks[0], -1).mean(0)
log(f'a softmax inside block 0 alone would give {", ".join(f"{s:.3f}" for s in lone.tolist())}: '
    'the denominator must run over every block')
OUT['eq4'] = EQ4

# ---------------------------------------------------------------- 3. Method 2: the library
log('')
log('== 3. The same attention with the library: past_key_values + torch scaled_dot_product_attention')
with torch.no_grad():
    ids = torch.tensor([PROMPT_A + OUT_A])
    lib = model(ids, use_cache=True)
    kv = lib.past_key_values                                    # contiguous: (1, heads, 10 tokens, 64) per layer
diffs = []
for l in range(LAYERS):
    q, o_paged = TRACE[l]
    K, V = kv.layers[l].keys, kv.layers[l].values
    o_lib = F.scaled_dot_product_attention(q[None, :, None], K, V)[0, :, 0]   # (H, d), one query, all 10 keys
    diffs.append((o_lib - o_paged).abs().max().item())
log(f'attention output of "forth", paged (Method 1) against SDPA on the contiguous cache (Method 2), 12 layers:')
log(f'  max difference {max(diffs):.1e} (layer {diffs.index(max(diffs))}), smallest {min(diffs):.1e}')
d_logits = (lib.logits[0, -1] - logits).abs().max().item()
log(f'logits after "forth": our paged engine against the library, max difference {d_logits:.1e}; '
    f'same top token: {lib.logits[0, -1].argmax().item() == logits.argmax().item()}')
OUT['check'] = {'attn_max': max(diffs), 'logits_max': d_logits}

# ---------------------------------------------------------------- 4. block sizes 1, 4, 16 with a random free list
log('')
log('== 4. Greedy generation of 48 tokens, our paged engine against model.generate (random free-block order)')
PROMPT = tok('The best way to learn a new language is').input_ids          # with OPT's start token </s>
NEW = 48
with torch.no_grad():
    ref = model.generate(torch.tensor([PROMPT]), max_new_tokens=NEW, do_sample=False,
                         output_logits=True, return_dict_in_generate=True)
ref_tokens = ref.sequences[0, len(PROMPT):].tolist()
log(f'prompt {len(PROMPT)} tokens; library text: {tok.decode(ref_tokens)[:70]!r}...')
OUT['blocksizes'] = {}
for B in (1, 4, 16):
    nb = math.ceil((len(PROMPT) + NEW) / B) + 4
    order = list(range(nb))
    random.Random(B).shuffle(order)
    cache = PagedKVCache(nb, B, order=order)
    lg = forward(PROMPT, 0, 'r', cache)
    toks, worst = [], 0.0
    for i in range(NEW):
        worst = max(worst, (lg - ref.logits[i][0]).abs().max().item())
        nxt = lg.argmax().item()
        toks.append(nxt)
        if i < NEW - 1:
            lg = forward([nxt], len(PROMPT) + i, 'r', cache)
    phys = [p for p, _ in cache.tables['r']]
    log(f'B = {B:>2}: {len(phys):>2} blocks, first physical blocks {phys[:6]}; same 48 tokens: {toks == ref_tokens}; '
        f'max logit difference {worst:.1e}')
    OUT['blocksizes'][B] = {'blocks': len(phys), 'same': toks == ref_tokens, 'max_diff': worst}

# ---------------------------------------------------------------- 5. Figure 7: two requests in one pool
log('')
log('== 5. Figure 7: requests A and B share one pool of 9 physical blocks (block size 4)')
PROMPT_B = tok('It was the best', add_special_tokens=False).input_ids
OUT_B = tok(' of times', add_special_tokens=False).input_ids
WORDS['B'] = [tok.decode(t).strip() for t in PROMPT_B + OUT_B]
pool7 = PagedKVCache(9, 4, order=ORDER)
lA = forward(PROMPT_A, 0, 'A', pool7)
lB = forward(PROMPT_B, 0, 'B', pool7)
log(f'iteration 1 (both prefills): A: {table_str(pool7, "A")}')
log(f'                             B: {table_str(pool7, "B")}')
ITERS = [{'A': [list(e) for e in pool7.tables['A']], 'B': [list(e) for e in pool7.tables['B']]}]
fedA, fedB = list(PROMPT_A), list(PROMPT_B)
for it, (ta, tb) in enumerate(zip(OUT_A[:2], OUT_B), start=2):
    lA = forward([ta], len(fedA), 'A', pool7); fedA.append(ta)
    lB = forward([tb], len(fedB), 'B', pool7); fedB.append(tb)
    log(f'iteration {it} (A writes {tok.decode(ta).strip()!r}, B writes {tok.decode(tb).strip()!r}): '
        f'A: {[p for p, _ in pool7.tables["A"]]}, B: {[p for p, _ in pool7.tables["B"]]}')
    ITERS.append({'A': [list(e) for e in pool7.tables['A']], 'B': [list(e) for e in pool7.tables['B']],
                  'wA': tok.decode(ta).strip(), 'wB': tok.decode(tb).strip()})
owner = {}
pool = words_in(pool7, owner, WORDS)
log('physical blocks:')
for b in range(9):
    log(f'  block {b}: ' + (f'{owner[b]}  ' + ' '.join(f'{w:>8}' for w in pool[b]) if b in pool else '   -'))
with torch.no_grad():
    refA = model(torch.tensor([fedA])).logits[0, -1]
    refB = model(torch.tensor([fedB])).logits[0, -1]
log(f'each request against the library run alone: A max difference {(refA - lA).abs().max():.1e}, '
    f'B max difference {(refB - lB).abs().max():.1e}')
log(f'free list before A ends: {pool7.alloc.free}')
pool7.free('A')
log(f'A ends; its blocks go back. free list now: {pool7.alloc.free}')
WORDS['C'] = [tok.decode(t).strip() for t in tok('Call me Ishmael. Some years ago', add_special_tokens=False).input_ids]
forward(tok('Call me Ishmael. Some years ago', add_special_tokens=False).input_ids, 0, 'C', pool7)
log(f'a new request C of {len(WORDS["C"])} tokens takes physical blocks {[p for p, _ in pool7.tables["C"]]}')
OUT['fig7'] = {'pool': pool, 'owner': owner, 'iters': ITERS, 'words': {k: WORDS[k] for k in 'ABC'},
               'C': [list(e) for e in pool7.tables['C']]}

# ---------------------------------------------------------------- 6. waste per request on real lengths
log('')
log('== 6. Waste per request at the end of generation, 2,000 Alpaca requests (prompt + answer, OPT tokenizer)')
alp = load_dataset('tatsu-lab/alpaca')['train']
rng = random.Random(0)
LENS = []
for i in rng.sample(range(len(alp)), 2000):
    r = alp[i]
    prompt = r['instruction'] + ('\n' + r['input'] if r['input'] else '')
    LENS.append(len(tok(prompt).input_ids) + len(tok(r['output']).input_ids) - 1)   # one start token
log(f'lengths: mean {sum(LENS) / len(LENS):.1f}, shortest {min(LENS)}, longest {max(LENS)} tokens; '
    f'{sum(LENS):,} tokens in all')


def waste_by_table(L, B):
    """Method 1: append L tokens one by one through a block table; count the empty slots at the end."""
    table = []
    for _ in range(L):
        if not table or table[-1] == B:
            table.append(0)                                     # a new block, only when the last one is full
        table[-1] += 1
    return len(table) * B - L


def waste_closed(L, B):
    """Method 2: ceil(L / B) blocks of B slots, minus the L slots in use."""
    return B * math.ceil(L / B) - L


log(f'{"block size":>10} {"max waste":>9} {"bound B-1":>9} {"mean waste":>10} {"slots held":>10} {"in use":>7}  methods agree')
WASTE = {}
for B in (1, 2, 4, 8, 16, 32, 64, 128):
    w1 = [waste_by_table(L, B) for L in LENS]
    w2 = [waste_closed(L, B) for L in LENS]
    held = sum(LENS) + sum(w1)
    WASTE[B] = {'max': max(w1), 'mean': sum(w1) / len(w1), 'held': held, 'use': 100 * sum(LENS) / held}
    log(f'{B:>10} {max(w1):>9} {B - 1:>9} {WASTE[B]["mean"]:>10.1f} {held:>10,} {WASTE[B]["use"]:>6.1f}%  {w1 == w2}')
held_c = 2048 * len(LENS)
log(f'{"contiguous":>10} {2048 - min(LENS):>9} {"":>9} {held_c / len(LENS) - sum(LENS) / len(LENS):>10.1f} '
    f'{held_c:>10,} {100 * sum(LENS) / held_c:>6.1f}%  (one chunk of 2,048 slots per request)')
PER_TOK_13B = 2 * 5120 * 40 * 2                                 # bytes of KV cache per token for OPT-13B
OUT['contig'] = {'held': held_c, 'use': 100 * sum(LENS) / held_c}
ROOM = int(12e9 // PER_TOK_13B)                                # Part 1's KV cache room: 12 GB of a 40 GB A100
fit16 = ROOM / (WASTE[16]['held'] / len(LENS))
log(f'OPT-13B with 12 GB of KV cache = {ROOM:,} slots: chunks of 2,048 fit {ROOM // 2048} requests; '
    f'blocks of 16 fit {fit16:.0f} requests of these lengths, even at their full length')
OUT['fit'] = {'room': ROOM, 'contig': ROOM // 2048, 'paged16': fit16}
OUT['waste'] = WASTE
OUT['lens'] = {'mean': sum(LENS) / len(LENS), 'min': min(LENS), 'max': max(LENS), 'total': sum(LENS)}

# ---------------------------------------------------------------- 7. the cost side of the block size
log('')
log('== 7. Block size against speed: our block-by-block attention, one query, a 2,048-token cache, one layer (CPU)')
T = 2048
Kc, Vc, qc = torch.randn(T, H, DH), torch.randn(T, H, DH), torch.randn(H, DH)
ref_o = F.scaled_dot_product_attention(qc[:, None], Kc.transpose(0, 1), Vc.transpose(0, 1))[:, 0]


def fastest(fn, repeats=7):
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


SPEED = {}
for B in (1, 4, 16, 64, 256):
    nb = T // B
    c = PagedKVCache(nb, B)
    c.K = c.K[:1].clone(); c.V = c.V[:1].clone()               # one layer is enough here
    order = list(range(nb)); random.Random(1).shuffle(order)
    table = [[p, B] for p in order]
    for j, p in enumerate(order):
        c.K[0, p], c.V[0, p] = Kc[j * B:(j + 1) * B], Vc[j * B:(j + 1) * B]
    o = paged_attention(qc, c, 0, table)
    ms = fastest(lambda: paged_attention(qc, c, 0, table)) * 1e3
    SPEED[B] = ms
    log(f'B = {B:>3}: {nb:>4} blocks, {ms:7.2f} ms; max difference from SDPA {(o - ref_o).abs().max():.1e}')
ms = fastest(lambda: F.scaled_dot_product_attention(qc[:, None], Kc.transpose(0, 1), Vc.transpose(0, 1))) * 1e3
log(f'contiguous SDPA, one call: {ms:.2f} ms')
log('fewer, larger blocks mean fewer separate fetches; but the last block of each request wastes up to B - 1 slots')
OUT['speed'] = SPEED
OUT['speed_sdpa'] = ms


# ---------------------------------------------------------------- 8. the lookup: paging's address split, for tokens
log('')
log('== 8. Finding a token through the block table, like an address through a page table')


def locate(t, table, B):
    """The address split of paging. t: a position (a token, or a byte address). table: physical number of each
    logical block (or page). Returns (logical block, offset, physical block, slot in the pool as one long row)."""
    j, off = t // B, t % B                                      # which block, and where inside it
    phys = table[j]                                             # one look into the table
    return j, off, phys, phys * B + off


log('the operating system (Background page): page size 4,096 bytes, page table [5, 2, 7, 0]')
j, off, phys, flat = locate(10000, [5, 2, 7, 0], 4096)
log(f'  address 10,000 -> page {j}, offset {off:,} -> frame {phys} -> physical address {flat:,}')
TAB_A = [p for p, _ in cacheA.tables['A']]
log(f'vLLM (request A of Figure 6): block size 4, block table {TAB_A}')
LOOK = []
worst = 0.0
for t, w in enumerate(WORDS['A']):
    j, off, phys, flat = locate(t, TAB_A, 4)
    for l in range(LAYERS):                                     # the key found by the lookup against the library's key
        worst = max(worst, (cacheA.K[l, phys, off] - kv.layers[l].keys[0, :, t]).abs().max().item(),
                    (cacheA.V[l, phys, off] - kv.layers[l].values[0, :, t]).abs().max().item())
    LOOK.append([t, w, j, off, phys, flat])
    if t in (0, 4, 7, 8, 9):
        log(f'  token {t} {w!r:>10} -> logical block {j}, offset {off} -> physical block {phys} -> slot {flat:>2} of the pool')
log(f'keys and values found this way against the library\'s contiguous cache, 10 tokens x 12 layers: '
    f'max difference {worst:.1e}')
OUT['lookup'] = {'tokens': LOOK, 'max_diff': worst}

# ---------------------------------------------------------------- 9. why splitting the softmax is exact
log('')
log('== 9. Eq. 4 by hand: the Background example (query (1, 0), three tokens) in blocks of 2')
qx = [1.0, 0.0]
Kx = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
Vx = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
blocks_x = [[0, 1], [2]]                                        # block 1 holds tokens 1 and 2, block 2 holds token 3
sc = [sum(a * b for a, b in zip(qx, k)) / math.sqrt(2) for k in Kx]          # q . k / sqrt(d), d = 2
ex = [math.exp(s) for s in sc]
part = [sum(ex[t] for t in blk) for blk in blocks_x]           # each block adds up its own exponentials
den = sum(part)                                                 # the one shared denominator
TINY = {'scores': sc, 'exps': ex, 'block_sums': part, 'den': den, 'blocks': []}
for b_, blk in enumerate(blocks_x):
    wts = [ex[t] / den for t in blk]
    contrib = [sum(wts[i] * Vx[t][c] for i, t in enumerate(blk)) for c in range(2)]
    log(f'block {b_ + 1}: scores {", ".join(f"{sc[t]:.3f}" for t in blk)}; exp {", ".join(f"{ex[t]:.3f}" for t in blk)}; '
        f'block sum {part[b_]:.3f}; weights {", ".join(f"{w:.3f}" for w in wts)}; adds ({contrib[0]:.3f}, {contrib[1]:.3f})')
    TINY['blocks'].append({'weights': wts, 'contrib': contrib})
o_x = [sum(blk['contrib'][c] for blk in TINY['blocks']) for c in range(2)]
lib_x = torch.softmax(torch.tensor(sc), 0) @ torch.tensor(Vx)
log(f'shared denominator {part[0]:.3f} + {part[1]:.3f} = {den:.3f}; output ({o_x[0]:.3f}, {o_x[1]:.3f}); '
    f'torch softmax over all three tokens: ({lib_x[0]:.3f}, {lib_x[1]:.3f})')
TINY['out'] = o_x
OUT['tiny'] = TINY


def merged_attention(q, cache, layer, table):
    """Each block alone keeps three things: its largest score m_j, its sum s_j of exp(score - m_j) and its
    unnormalised output u_j. Merging rescales every block to the largest m and divides once at the end."""
    scale = 1 / math.sqrt(DH)
    parts = []
    for phys, filled in table:
        s = torch.einsum('hd,thd->ht', q, cache.K[layer, phys, :filled]) * scale       # (H, filled)
        m = s.max(-1).values                                                           # (H,)
        e = torch.exp(s - m[:, None])
        parts.append((m, e.sum(-1), torch.einsum('ht,thd->hd', e, cache.V[layer, phys, :filled])))
    M = torch.stack([m for m, _, _ in parts]).max(0).values
    S = sum(s_ * torch.exp(m - M) for m, s_, _ in parts)                               # the shared denominator
    U = sum(u * torch.exp(m - M)[:, None] for m, _, u in parts)
    return U / S[:, None]


log('blocks finished one at a time and merged at the end, query of "forth", 12 layers of OPT-125M:')
mdiff = 0.0
for l in range(LAYERS):
    q, o_paged = TRACE[l]
    o_lib = F.scaled_dot_product_attention(q[None, :, None], kv.layers[l].keys, kv.layers[l].values)[0, :, 0]
    mdiff = max(mdiff, (merged_attention(q, cacheA, l, cacheA.tables['A']) - o_lib).abs().max().item())
log(f'  merged against SDPA on the contiguous cache: max difference {mdiff:.1e}')
OUT['merge_max'] = mdiff

save('part3', OUT)
