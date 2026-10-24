"""Part 4 of vLLM, explained: sharing, scheduling and many GPUs (§4.4 to §4.6, §5), run on a laptop.

What it shows (every number is printed and saved to results/part4_stdout.txt and results/part4.json):
  1. A KV block manager from scratch: a pool of physical blocks, a block table and a reference count per block,
     and the three calls of §5.2: fork, append and free. Copy-on-write happens inside append.
  2. Parallel sampling (Figure 8): 4 samples share the blocks of one prompt. Method 1 is our paged loop on
     OPT-125M; Method 2 is the library's model.generate with the same random seed. The tokens must agree.
  3. Beam search (Figure 9): 4 beams fork, append and free blocks at every step. Checked against model.generate.
  4. Memory saved by sharing (as in Figure 15) for 2, 4 and 6 samples or beams, block size 16.
  5. Shared prefix (Figure 10): the blocks of a few-shot prefix are computed once and mapped by two requests.
  6. Scheduling and preemption (§4.5): a first-come-first-served scheduler that preempts the latest request when
     blocks run out, with swapping or recomputation, on Alpaca request lengths (a simulation in steps).
  7. Swapping against recomputation on the real model: both give back the same cache; what each one costs here.
  8. Tensor parallelism (§4.6): one OPT-125M layer split over 2 simulated GPUs, Megatron style. The all-reduce
     (a sum) of the two halves must equal the unsplit layer.
  9. Two kernels of §5.1 in PyTorch: the block write (one scatter for all new tokens) and the batched block copy.
Model: facebook/opt-125m, FP32 on the CPU for exact checks (2 to 5, 7, 8) and FP16 on Apple MPS for timing (7, 9).
Run: python vllm_part4.py   (about 3 minutes on an M-series Mac)"""
import math, random, statistics
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from common import Log, save, timer

log = Log('part4')
NAME = 'facebook/opt-125m'
GPU = 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = {}

tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32).eval()
cfg = model.config
L, H = cfg.num_hidden_layers, cfg.num_attention_heads              # 12 layers, 12 heads
D = cfg.hidden_size // H                                            # 64 numbers per head
V, EOS = cfg.vocab_size, cfg.eos_token_id


def sync():
    if GPU == 'mps':
        torch.mps.synchronize()


def fastest(fn, repeats=5):
    """Run fn several times; return the fastest wall time in seconds (other work on the laptop only adds time)."""
    best = 1e9
    for _ in range(repeats):
        sync(); clock = timer(); fn(); sync()
        best = min(best, clock())
    return best


# ---------------------------------------------------------------- 1. the block manager: fork, append, free
class BlockManager:
    """vLLM's KV cache manager in miniature. One physical block holds the keys and values of B tokens for every
    layer and head. Each sequence has a block table (a list of physical block numbers) and a length."""

    def __init__(self, num_blocks, block_size, dtype=torch.float32, device='cpu', seed=0):
        self.B = block_size
        self.pool = torch.zeros(num_blocks, L, 2, H, block_size, D, dtype=dtype, device=device)
        order = list(range(num_blocks))
        random.Random(seed).shuffle(order)                  # hand out blocks in a scattered order, as in the figures
        self.free_blocks = order
        self.ref = [0] * num_blocks                         # reference count: how many sequences map each block
        self.table, self.length = {}, {}
        self.cow_copies = 0

    def _allocate(self):
        if not self.free_blocks:
            raise MemoryError('no free block')
        b = self.free_blocks.pop(0)
        self.ref[b] = 1
        return b

    def new(self, seq):
        self.table[seq], self.length[seq] = [], 0

    def fork(self, parent, child):
        """fork: a new sequence that maps every block of its parent. Nothing is copied; reference counts go up."""
        self.table[child], self.length[child] = list(self.table[parent]), self.length[parent]
        for b in self.table[child]:
            self.ref[b] += 1

    def append(self, seq, kv):
        """append: store the keys and values of one new token. kv: (layers, 2, heads, head size)."""
        t, pos = self.table[seq], self.length[seq]
        if pos % self.B == 0:                               # the last block is full (or there is none): take a new one
            t.append(self._allocate())
        elif self.ref[t[-1]] > 1:                           # the last block is shared: copy-on-write
            new = self._allocate()
            self.pool[new].copy_(self.pool[t[-1]])          # copy the one block, then write into the copy
            self.ref[t[-1]] -= 1
            t[-1] = new
            self.cow_copies += 1
        self.pool[t[-1], :, :, :, pos % self.B] = kv
        self.length[seq] = pos + 1

    def free(self, seq):
        """free: drop a sequence. A block goes back to the free list when no sequence maps it any more."""
        for b in self.table.pop(seq):
            self.ref[b] -= 1
            if self.ref[b] == 0:
                self.free_blocks.append(b)
        del self.length[seq]

    def gather(self, seq):
        """Read a sequence's blocks in block-table order: (layers, 2, heads, tokens, head size)."""
        blocks = self.pool[self.table[seq]]                 # (blocks, layers, 2, heads, B, head size)
        return blocks.permute(1, 2, 3, 0, 4, 5).reshape(L, 2, H, -1, D)[:, :, :, :self.length[seq]]

    def used(self):
        return sum(r > 0 for r in self.ref)                 # physical blocks in use, shared blocks counted once

    def unshared(self):
        return sum(math.ceil(n / self.B) for n in self.length.values())   # blocks if every sequence had its own


@torch.no_grad()
def extend(mgr, seqs, tokens):
    """Run the model on new tokens of one or more sequences of equal length. The earlier keys and values are read
    from the blocks; the new ones are stored with append. tokens: list (one list of ids per sequence).
    Returns the logits of the last new token of each sequence: (sequences, vocabulary)."""
    n = mgr.length[seqs[0]]
    past = DynamicCache()
    if n:
        caches = [mgr.gather(s) for s in seqs]
        for l in range(L):
            past.update(torch.stack([c[l, 0] for c in caches]), torch.stack([c[l, 1] for c in caches]), l)
    ids = torch.tensor(tokens)
    out = model(input_ids=ids, past_key_values=past, use_cache=True,
                attention_mask=torch.ones(len(seqs), n + ids.shape[1], dtype=torch.long))
    for i, s in enumerate(seqs):
        for p in range(n, n + ids.shape[1]):
            kv = torch.stack([torch.stack([out.past_key_values.layers[l].keys[i, :, p],
                                           out.past_key_values.layers[l].values[i, :, p]]) for l in range(L)])
            mgr.append(s, kv)
    return out.logits[:, -1]


def words_of(mgr, seq, ids):
    """The text of each block of a sequence, for the figures."""
    return [[tok.decode([x]) for x in ids[j * mgr.B:(j + 1) * mgr.B]] for j in range(len(mgr.table[seq]))]


# ---------------------------------------------------------------- 2. parallel sampling (Figure 8)
@torch.no_grad()
def parallel_sampling(prompt, n, T, B, seed, trace=None):
    """Method 1: prompt phase once, fork n samples, then sample and append. The eos token is masked so every sample
    has T tokens (the library does the same with min_new_tokens=T)."""
    mgr = BlockManager(512, B)
    mgr.new('prompt')
    logits = extend(mgr, ['prompt'], [prompt])
    seqs = [f's{i}' for i in range(n)]
    for s in seqs:
        mgr.fork('prompt', s)
    mgr.free('prompt')
    if trace is not None:
        trace['before'] = {s: list(mgr.table[s]) for s in seqs}
        trace['ref_before'] = {b: mgr.ref[b] for b in mgr.table[seqs[0]]}
        trace['blocks_text'] = words_of(mgr, seqs[0], prompt)
    logits = logits.expand(n, -1)
    torch.manual_seed(seed)
    outs, all_logits, steps = [[] for _ in seqs], [logits.clone()], []
    for t in range(T):
        scores = logits.clone()
        scores[:, EOS] = -float('inf')
        nxt = torch.multinomial(torch.softmax(scores, -1), 1)[:, 0].tolist()
        for o, x in zip(outs, nxt):
            o.append(x)
        if t == T - 1:
            break
        logits = extend(mgr, seqs, [[x] for x in nxt])
        all_logits.append(logits.clone())
        steps.append((mgr.used(), mgr.unshared()))
        if trace is not None and t == 0:
            trace['after'] = {s: list(mgr.table[s]) for s in seqs}
            trace['cow_after_first'] = mgr.cow_copies
            trace['first'] = [tok.decode([x]) for x in nxt]
    return dict(outs=outs, logits=all_logits, steps=steps, cow=mgr.cow_copies, mgr=mgr, seqs=seqs)


log('== 1. Parallel sampling (Figure 8): 4 samples share the blocks of one prompt, block size 4')
PROMPT8 = tok('Four score and seven years ago', return_tensors='pt').input_ids[0].tolist()
N8, T8, B8, SEED = 4, 24, 4, 3
fig8 = {}
r = parallel_sampling(PROMPT8, N8, T8, B8, SEED, fig8)
log(f'prompt of {len(PROMPT8)} tokens (with the start token </s>) in {len(fig8["blocks_text"])} blocks: '
    + ' | '.join(' '.join(w.strip() or '</s>' for w in blk) for blk in fig8['blocks_text']))
log(f'after fork: every sample maps blocks {fig8["before"]["s0"]}; reference counts {fig8["ref_before"]}')
log(f'after the first new token: block tables {list(fig8["after"].values())}; copy-on-write copies: {fig8["cow_after_first"]}')
torch.manual_seed(SEED)
with torch.no_grad():
    g = model.generate(torch.tensor([PROMPT8]), do_sample=True, num_return_sequences=N8, max_new_tokens=T8,
                       min_new_tokens=T8, top_k=0, top_p=1.0, pad_token_id=tok.pad_token_id)
lib = [row[len(PROMPT8):].tolist() for row in g]
same_tokens = lib == r['outs']
full = [torch.tensor([PROMPT8 + o]) for o in r['outs']]
with torch.no_grad():
    ref_logits = [model(x).logits[0, len(PROMPT8) - 1:-1] for x in full]          # one pass per sample, no cache
diff8 = max((r['logits'][t][i] - ref_logits[i][t]).abs().max().item() for i in range(N8) for t in range(T8))
for i, o in enumerate(r['outs']):
    log(f'  sample {i + 1}: {tok.decode(o)!r}')
log(f'Method 1 (paged, shared prompt) and Method 2 (model.generate, same seed): same {N8 * T8} tokens: {same_tokens}')
log(f'max difference of the logits against one plain pass per sample: {diff8:.1e}')
u, s = r['steps'][-1]
log(f'at the end: {u} physical blocks with sharing, {s} without; copy-on-write copies in total: {r["cow"]}')
OUT['fig8'] = {**fig8, 'samples': [tok.decode(o) for o in r['outs']], 'same': same_tokens, 'diff': diff8,
               'steps': r['steps'], 'cow': r['cow'], 'prompt_len': len(PROMPT8)}


# ---------------------------------------------------------------- 3. beam search (Figure 9)
@torch.no_grad()
def beam_search(prompt, k, T, B):
    """Method 1: beam search on the block manager. At each step every new beam is a fork of its parent, the old
    beams are freed, and the new token is appended (copy-on-write when it lands in a shared block)."""
    mgr = BlockManager(1024, B)
    mgr.new('b0')
    logits = extend(mgr, ['b0'], [prompt])
    beams, scores, toks, nid = ['b0'], torch.zeros(1), {'b0': []}, 0
    steps, contiguous_copy, history = [], 0, []
    for t in range(T):
        logp = torch.log_softmax(logits, -1)
        logp[:, EOS] = -float('inf')                        # the library does this with min_new_tokens=T
        top = torch.topk((scores[:, None] + logp).flatten(), k)
        parents, words = (top.indices // V).tolist(), (top.indices % V).tolist()
        new = []
        for i, (p, w) in enumerate(zip(parents, words)):
            nid += 1
            c = f'b{nid}'
            mgr.fork(beams[p], c)
            toks[c] = toks[beams[p]] + [w]
            new.append(c)
            if p != i:                                      # a contiguous cache would copy the parent's whole cache
                contiguous_copy += mgr.length[beams[p]]
        for b in beams:
            mgr.free(b)
        history.append(parents)
        beams, scores = new, top.values
        if t == T - 1:
            break
        logits = extend(mgr, beams, [[toks[b][-1]] for b in beams])
        steps.append((mgr.used(), mgr.unshared()))
    tables = [list(mgr.table[b]) for b in beams]
    return dict(outs=[toks[b] for b in beams], steps=steps, cow=mgr.cow_copies, tables=tables, history=history,
                contiguous_copy=contiguous_copy, refs={b: mgr.ref[b] for t_ in tables for b in t_}, mgr=mgr)


log('')
log('== 2. Beam search (Figure 9): 4 beams fork, append and free blocks at every step, block size 4')
K9, T9 = 4, 24
r9 = beam_search(PROMPT8, K9, T9, B8)
with torch.no_grad():
    g = model.generate(torch.tensor([PROMPT8]), num_beams=K9, num_return_sequences=K9, do_sample=False,
                       max_new_tokens=T9, min_new_tokens=T9, pad_token_id=tok.pad_token_id)
lib9 = [row[len(PROMPT8):].tolist() for row in g]
for i, o in enumerate(r9['outs']):
    log(f'  beam {i + 1}: {tok.decode(o)!r}')
log(f'Method 1 (paged beam search) and Method 2 (model.generate, num_beams=4): same {K9 * T9} tokens: {lib9 == r9["outs"]}')
log(f'final block tables: {r9["tables"]}')
shared_all = [b for b in r9['tables'][0] if all(b in t for t in r9['tables'])]
log(f'blocks mapped by all 4 beams: {shared_all}; physical blocks in use {r9["steps"][-1][0]}, '
    f'without sharing {r9["steps"][-1][1]}')
log(f'copied during the search: {r9["cow"]} blocks by copy-on-write = {r9["cow"] * B8} token slots; a contiguous '
    f'cache that keeps beam i in slot i copies {r9["contiguous_copy"]} token slots')
OUT['fig9'] = {'tables': r9['tables'], 'refs': r9['refs'], 'steps': r9['steps'], 'cow': r9['cow'],
               'contiguous_copy': r9['contiguous_copy'], 'same': lib9 == r9['outs'], 'history': r9['history'],
               'beams': [tok.decode(o) for o in r9['outs']], 'prompt_len': len(PROMPT8)}


# ---------------------------------------------------------------- 4. memory saved by sharing (Figure 15)
log('')
log('== 3. Memory saved by sharing blocks, block size 16 (the vLLM default), prompt of the Figure 10 prefix')
PREFIX = ('Translate English to French:\nsea otter => loutre de mer\npeppermint => menthe poivrée\n'
          'plush girafe => girafe en peluche\n')
PRE = tok(PREFIX).input_ids                                  # with the start token
T15 = 48


def saving(steps):
    return 100 * statistics.mean(1 - u / s for u, s in steps)


sweep = {'sampling': {}, 'beam': {}}
for n in (2, 4, 6):
    sweep['sampling'][n] = saving(parallel_sampling(PRE, n, T15, 16, SEED)['steps'])
for k in (2, 4, 6):
    sweep['beam'][k] = saving(beam_search(PRE, k, T15, 16)['steps'])
log(f'prompt of {len(PRE)} tokens, {T15} new tokens; saving = blocks saved by sharing / blocks without sharing, '
    'averaged over the steps')
log('parallel sampling, 2 / 4 / 6 samples: ' + ' / '.join(f'{v:.1f}%' for v in sweep['sampling'].values()))
log('beam search, width 2 / 4 / 6:          ' + ' / '.join(f'{v:.1f}%' for v in sweep['beam'].values()))
OUT['sweep'] = sweep


# ---------------------------------------------------------------- 5. shared prefix (Figure 10)
log('')
log('== 4. Shared prefix (Figure 10): the prefix is computed once and two requests map its blocks')
mgr = BlockManager(256, 4)
mgr.new('prefix')
extend(mgr, ['prefix'], [PRE])                               # done once, ahead of time, by the service
prefix_blocks = list(mgr.table['prefix'])
res5 = []
for task in ('cheese =>', 'I love you =>'):
    ids = tok(task, add_special_tokens=False).input_ids
    mgr.fork('prefix', task)
    logits = extend(mgr, [task], [ids])                      # the prompt phase runs on the task input only
    with torch.no_grad():
        full = model(torch.tensor([PRE + ids])).logits[0, -1]
    out = []
    for _ in range(8):                                      # greedy decoding on the paged cache
        x = int(logits[0].argmax())
        out.append(x)
        if x == EOS:
            break
        logits = extend(mgr, [task], [[x]])
    with torch.no_grad():
        g = model.generate(torch.tensor([PRE + ids]), do_sample=False, max_new_tokens=8,
                           pad_token_id=tok.pad_token_id)[0, len(PRE) + len(ids):].tolist()
    shared = sum(1 for b in mgr.table[task] if b in prefix_blocks)
    res5.append(dict(task=task, n=len(ids), out=tok.decode(out[:len(g)]), same=out[:len(g)] == g,
                     shared=shared, own=len(mgr.table[task]) - shared))
    log(f'  {task!r}: {len(ids)} task tokens computed instead of {len(PRE) + len(ids)}; maps {shared} prefix blocks, '
        f'owns {len(mgr.table[task]) - shared}; greedy output {tok.decode(out[:len(g)])!r}; same as model.generate '
        f'on the full prompt: {out[:len(g)] == g}')
log(f'prefix of {len(PRE)} tokens in {len(prefix_blocks)} blocks; its last block is partly full, so each request '
    f'copied it once: {mgr.cow_copies} copy-on-write copies')
OUT['prefix'] = {'len': len(PRE), 'blocks': len(prefix_blocks), 'requests': res5, 'cow': mgr.cow_copies}


# ---------------------------------------------------------------- 6. scheduling and preemption (§4.5)
log('')
log('== 5. First-come-first-served scheduling with preemption (simulation in steps, Alpaca request lengths)')
alp = load_dataset('tatsu-lab/alpaca')['train']
rng = random.Random(0)
REQ = []
for i in rng.sample(range(len(alp)), 400):
    p = len(tok(alp[i]['instruction'] + '\n' + alp[i]['input']).input_ids)
    o = len(tok(alp[i]['output']).input_ids) - 1
    if o > 0:
        REQ.append((p, o))
REQ = REQ[:200]
GPU_BLOCKS, BS, GAP = 64, 16, 1                               # 64 blocks of 16 tokens = 1,024 token slots


def serve(policy):
    """One iteration per loop: admit, grow, preempt, write one token per running request.
    policy: 'swap' copies a preempted request's blocks to CPU memory; 'recompute' frees them and later runs
    prompt + generated tokens again as one prompt."""
    blocks = lambda n: math.ceil(n / BS)
    arrive = [i * GAP for i in range(len(REQ))]
    gen = [0] * len(REQ)                                    # tokens written so far
    running, waiting, preempted, done = [], [], [], 0
    free, cpu, cpu_max = GPU_BLOCKS, 0, 0
    st = dict(preemptions=0, swap_out=0, swap_in=0, recomputed=0, batch=[], free=[], preempt_steps=[])
    t, nxt = 0, 0
    while done < len(REQ):
        while nxt < len(REQ) and arrive[nxt] <= t:
            waiting.append(nxt); nxt += 1
        # 1. bring back preempted requests first, oldest first; no new request while any is preempted
        while preempted and blocks(REQ[preempted[0]][0] + gen[preempted[0]]) <= free:
            r = preempted.pop(0)
            need = blocks(REQ[r][0] + gen[r])
            free -= need
            if policy == 'swap':
                st['swap_in'] += need; cpu -= need
            else:
                st['recomputed'] += REQ[r][0] + gen[r]
            running.append(r)
        while not preempted and waiting and blocks(REQ[waiting[0]][0]) <= free:
            r = waiting.pop(0)
            free -= blocks(REQ[r][0])                       # the prompt's blocks
            running.append(r)
        running.sort()
        # 2. every running request writes one token; it may need one more block. If none is free, preempt the
        #    request that arrived last (all of its blocks), as often as needed.
        for r in list(running):
            if r not in running:
                continue
            n = REQ[r][0] + gen[r]                          # tokens stored so far
            if n % BS == 0:                                 # the last block is full
                while free == 0:
                    victim = running.pop()                   # the latest arrival
                    hold = blocks(REQ[victim][0] + gen[victim])
                    free += hold
                    st['preemptions'] += 1
                    st['preempt_steps'].append(t)
                    if policy == 'swap':
                        st['swap_out'] += hold; cpu += hold; cpu_max = max(cpu_max, cpu)
                    preempted.append(victim); preempted.sort()
                    if victim == r:
                        break
                if r not in running:
                    continue
                free -= 1
        st['batch'].append(len(running))
        for r in list(running):
            gen[r] += 1
            if gen[r] == REQ[r][1]:
                running.remove(r); done += 1
                free += blocks(REQ[r][0] + gen[r])
        st['free'].append(free)
        t += 1
    st.update(steps=t, cpu_max=cpu_max, tokens=sum(gen), complete=gen == [o for _, o in REQ])
    return st


KV13 = 2 * 5120 * 40 * 2                                     # OPT-13B: 819,200 bytes of KV cache per token
KV125 = 2 * cfg.hidden_size * L * 2                          # OPT-125M in FP16: 36,864 bytes per token
sim = {p: serve(p) for p in ('swap', 'recompute')}
log(f'{len(REQ)} requests, one per step; prompt mean {statistics.mean(p for p, _ in REQ):.1f} tokens, output mean '
    f'{statistics.mean(o for _, o in REQ):.1f}; GPU memory: {GPU_BLOCKS} blocks of {BS} tokens')
for p, s in sim.items():
    log(f'{p:<9}: {s["steps"]:,} steps, mean batch {statistics.mean(s["batch"]):.1f}, {s["preemptions"]} preemptions; '
        f'every request got all its tokens: {s["complete"]}')
sw, rc = sim['swap'], sim['recompute']
log(f'swap     : {sw["swap_out"]} blocks out, {sw["swap_in"]} in; at most {sw["cpu_max"]} blocks in CPU memory '
    f'(never more than the {GPU_BLOCKS} GPU blocks: {sw["cpu_max"] <= GPU_BLOCKS})')
log(f'           bytes moved, OPT-13B: {(sw["swap_out"] + sw["swap_in"]) * BS * KV13 / 1e9:.2f} GB; '
    f'OPT-125M: {(sw["swap_out"] + sw["swap_in"]) * BS * KV125 / 1e6:.1f} MB')
log(f'recompute: {rc["recomputed"]:,} tokens computed again, in {rc["preemptions"]} prompt passes')
OUT['sched'] = {p: {k: v for k, v in s.items()} for p, s in sim.items()}
OUT['sched_cfg'] = {'blocks': GPU_BLOCKS, 'bs': BS, 'n': len(REQ)}


# ---------------------------------------------------------------- 7. swapping against recomputation, real model
log('')
log('== 6. Swapping against recomputation on the real model (OPT-125M)')
P7 = tok('Four score and seven years ago our fathers brought forth on this continent', return_tensors='pt').input_ids
T7 = 200
with torch.no_grad():
    g = model.generate(P7, do_sample=False, max_new_tokens=T7, min_new_tokens=T7, return_dict_in_generate=True,
                       pad_token_id=tok.pad_token_id)
seq7 = g.sequences                                           # prompt + 200 tokens; the cache holds all but the last
kv_inc = torch.stack([torch.stack([l.keys[0], l.values[0]]) for l in g.past_key_values.layers])
with torch.no_grad():
    out = model(seq7[:, :-1], use_cache=True)                # recomputation: one prompt pass over everything
kv_rec = torch.stack([torch.stack([l.keys[0], l.values[0]]) for l in out.past_key_values.layers])
with torch.no_grad():
    nxt_inc = int(model(seq7[:, -1:], past_key_values=g.past_key_values).logits[0, -1].argmax())
    nxt_rec = int(model(seq7[:, -1:], past_key_values=out.past_key_values).logits[0, -1].argmax())
log(f'a request with a {P7.shape[1]}-token prompt and {T7} written tokens: {kv_inc.shape[3]} tokens of KV cache')
log(f'recomputed in one pass against built one token at a time: max difference {((kv_rec - kv_inc).abs().max()):.1e}; '
    f'next token after recomputation the same: {nxt_rec == nxt_inc}')

gm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16).to(GPU).eval()
ids7 = seq7.to(GPU)
NT = kv_inc.shape[3]


@torch.no_grad()
def decode_original():
    """How the cache was first built: one pass for the prompt, then one pass per written token."""
    c = DynamicCache()
    gm(ids7[:, :P7.shape[1]], past_key_values=c)
    for p in range(P7.shape[1], NT):
        gm(ids7[:, p:p + 1], past_key_values=c)


@torch.no_grad()
def recompute():
    gm(ids7[:, :NT], use_cache=True)


kv_gpu = kv_inc.half().to(GPU)


def swap_times(B, kv=kv_gpu):
    """Copy a request's KV cache, cut into blocks of B tokens, from GPU memory to CPU memory and back, one copy
    per block (as cudaMemcpyAsync would do per block). Returns (out seconds, in seconds, round trip exact)."""
    n = math.ceil(kv.shape[3] / B)
    pad = torch.zeros(L, 2, H, n * B - kv.shape[3], D, dtype=kv.dtype, device=GPU)
    blocks = torch.cat([kv, pad], 3).reshape(L, 2, H, n, B, D).permute(3, 0, 1, 2, 4, 5).contiguous()
    cpu = torch.empty(blocks.shape, dtype=blocks.dtype)
    back = torch.empty_like(blocks)

    def out_():
        for j in range(n):
            cpu[j].copy_(blocks[j])

    def in_():
        for j in range(n):
            back[j].copy_(cpu[j])
    return fastest(out_), fastest(in_), torch.equal(back, blocks)


t_orig = fastest(decode_original, 3)
t_rec = fastest(recompute)
t_out, t_in, exact = swap_times(16)
nbytes = NT * KV125
log(f'laptop GPU ({GPU}), FP16, fastest of several runs; {NT} tokens = {nbytes / 1e6:.2f} MB of KV cache')
log(f'  building it the first time, one pass per token: {1000 * t_orig:7.1f} ms')
log(f'  recomputation, one prompt pass:                 {1000 * t_rec:7.1f} ms')
log(f'  swap out + swap in, blocks of 16 tokens:        {1000 * (t_out + t_in):7.1f} ms; round trip exact: {exact}')
log(f'  recomputation took {t_orig / t_rec:.0f} times less time than building the cache one token at a time')
sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
sweep7 = {}
for B in sizes:
    a, b, _ = swap_times(B)
    sweep7[B] = (1000 * a, 1000 * b)
log(f'swap time against block size (the same {NT} tokens; one copy per block):')
log('  block size ' + ' '.join(f'{B:>6}' for B in sizes))
log('  out ms     ' + ' '.join(f'{sweep7[B][0]:>6.2f}' for B in sizes))
log('  in ms      ' + ' '.join(f'{sweep7[B][1]:>6.2f}' for B in sizes))
log(f'  recompute  {1000 * t_rec:>6.2f} ms at every block size (it does not read the blocks)')
log(f'  block size 1 means {NT} copies each way: {sweep7[1][0] / NT:.2f} ms per copy out; block size 256 means 1 copy')
OUT['swap'] = {'tokens': NT, 'mb': nbytes / 1e6, 'orig_ms': 1000 * t_orig, 'rec_ms': 1000 * t_rec,
               'swap16_ms': 1000 * (t_out + t_in), 'sweep': sweep7}


# ---------------------------------------------------------------- 8. tensor parallelism (§4.6)
log('')
log('== 7. Tensor parallelism (§4.6): one OPT-125M layer split over 2 simulated GPUs, 6 heads each')
layer = model.model.decoder.layers[0]
attn = layer.self_attn
seen = {}
hook = attn.register_forward_hook(lambda m, a, kw, o: seen.update(x=kw['hidden_states'], y=o[0]), with_kwargs=True)
x_ids = tok('The paper splits every attention layer by its heads.', return_tensors='pt').input_ids
with torch.no_grad():
    hs = model(x_ids, output_hidden_states=True).hidden_states
hook.remove()
Tn = x_ids.shape[1]


@torch.no_grad()
def attention_shard(x, heads):
    """One GPU's work: the query, key and value rows of its own heads (column split), attention for those heads,
    and its columns of the output matrix (row split). The result is a partial sum, without the output bias."""
    rows = slice(heads.start * D, heads.stop * D)
    q = (x @ attn.q_proj.weight[rows].T + attn.q_proj.bias[rows]) * attn.scaling
    k = x @ attn.k_proj.weight[rows].T + attn.k_proj.bias[rows]
    v = x @ attn.v_proj.weight[rows].T + attn.v_proj.bias[rows]
    h = len(heads)
    q, k, v = (t.view(-1, h, D).transpose(0, 1) for t in (q, k, v))          # (heads, tokens, head size)
    s = q @ k.transpose(1, 2) + torch.full((Tn, Tn), -float('inf')).triu(1)   # causal mask
    o = (torch.softmax(s, -1) @ v).transpose(0, 1).reshape(-1, h * D)
    return o @ attn.out_proj.weight[:, rows].T, k                              # partial output, this GPU's keys


@torch.no_grad()
def mlp_shard(x, part, n=2):
    """The feed-forward layer split the same way: half of fc1's rows, then the matching half of fc2's columns."""
    f = layer.fc1.out_features // n
    rows = slice(part * f, (part + 1) * f)
    return torch.relu(x @ layer.fc1.weight[rows].T + layer.fc1.bias[rows]) @ layer.fc2.weight[:, rows].T


x = seen['x'][0]
parts = [attention_shard(x, range(0, 6)), attention_shard(x, range(6, 12))]
allreduce = parts[0][0] + parts[1][0] + attn.out_proj.bias                    # all-reduce = sum over the GPUs
d_attn = (allreduce - seen['y'][0]).abs().max().item()
log(f'{Tn} tokens; GPU 0 has heads 0-5, GPU 1 has heads 6-11; each computes a partial output of shape '
    f'{tuple(parts[0][0].shape)}')
log(f'all-reduce (sum of the 2 partial outputs) against the unsplit attention of the library: max difference {d_attn:.1e}')
with torch.no_grad():
    x0 = hs[0][0]
    h = x0 + sum(attention_shard(layer.self_attn_layer_norm(x0), hd)[0] for hd in (range(0, 6), range(6, 12))) \
        + attn.out_proj.bias                                                   # all-reduce 1
    y = h + sum(mlp_shard(layer.final_layer_norm(h), p) for p in (0, 1)) + layer.fc2.bias     # all-reduce 2
d_layer = (y - hs[1][0]).abs().max().item()
log(f'the whole layer (attention + feed-forward, 2 all-reduces) against the library layer: max difference {d_layer:.1e}')
kfull = torch.cat([parts[0][1], parts[1][1]])
with torch.no_grad():
    kref = (x @ attn.k_proj.weight.T + attn.k_proj.bias).view(-1, H, D).transpose(0, 1)
log(f'KV cache per GPU: keys of shape {tuple(parts[0][1].shape)} (6 heads, all {Tn} positions); the two halves '
    f'together equal the full keys: {torch.allclose(kfull, kref, atol=1e-6)}')
log(f'per GPU and per token: {KV125 // 2:,} bytes of KV cache instead of {KV125:,} (FP16); the same block table serves both')
log(f'data sent by each all-reduce: {Tn} x {cfg.hidden_size} numbers; 2 all-reduces per layer, {2 * L} per pass')
OUT['tp'] = {'attn_diff': d_attn, 'layer_diff': d_layer, 'tokens': Tn}


# ---------------------------------------------------------------- 9. two kernels of §5.1
log('')
log('== 8. Kernels of §5.1 in PyTorch, on the laptop GPU (FP16, one layer\'s key cache: blocks x 16 x 12 x 64)')
NB, B9 = 2048, 16
torch.manual_seed(0)
cache = torch.zeros(NB * B9, H, D, dtype=torch.float16, device=GPU)            # slot = block * 16 + offset
new = torch.randn(256, H, D, dtype=torch.float16, device=GPU)                  # 256 new tokens of a batch
tables = torch.randperm(NB)[:256 * 4].view(256, 4)                             # one block table per sequence
lens = torch.randint(1, 4 * B9, (256,))
slots = (tables[torch.arange(256), lens // B9] * B9 + lens % B9).to(GPU)       # where each new token goes


def write_loop():
    for i in range(256):
        cache[slots[i]] = new[i]


def write_fused():
    cache.index_copy_(0, slots, new)


write_loop(); a = cache.clone(); cache.zero_(); write_fused()
t_loop, t_fused = fastest(write_loop), fastest(write_fused)
log(f'block write of 256 new tokens: one copy per token {1000 * t_loop:.2f} ms ({256} launches), one scatter '
    f'{1000 * t_fused:.3f} ms (1 launch); same cache: {torch.equal(a, cache)}; {t_loop / t_fused:.0f}x')
perm = torch.randperm(NB)
src, dst = perm[:128].to(GPU), perm[128:256].to(GPU)             # 128 shared blocks and their new copies
blocks = cache.view(NB, B9, H, D)


def copy_loop():
    for s_, d_ in zip(src.tolist(), dst.tolist()):
        blocks[d_].copy_(blocks[s_])


def copy_fused():
    blocks.index_copy_(0, dst, blocks[src])


b0 = blocks.clone(); copy_loop(); a = blocks.clone(); blocks.copy_(b0); copy_fused()
same = torch.equal(a, blocks)
t_cl, t_cf = fastest(copy_loop), fastest(copy_fused)
log(f'copy-on-write of 128 scattered blocks: one copy per block {1000 * t_cl:.2f} ms, one batched copy '
    f'{1000 * t_cf:.3f} ms; same result: {same}; {t_cl / t_cf:.1f}x')
OUT['kernels'] = {'write_loop': 1000 * t_loop, 'write_fused': 1000 * t_fused, 'copy_loop': 1000 * t_cl,
                  'copy_fused': 1000 * t_cf}
log('')
log('timings on a shared laptop are noisy; each is the fastest of several runs')
save('part4', OUT)
