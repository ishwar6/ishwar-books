"""Part 6 of vLLM, explained: build it from scratch. Runs the minivllm package (code/papers/vllm/minivllm) on
the real weights of OPT-125M and checks it against the transformers library.

What it shows (every number is printed and saved to results/part6_stdout.txt and results/part6.json):
  1. The package: the files and their sizes.
  2. Block tables (§4.2, §4.3, Figures 6 and 7): two real prompts with block size 4 in a 9-block cache; the block
     table of each after the prompt step and after two decoding steps.
  3. PagedAttention, Eq. 4 (§4.1): Method 1 our block-by-block loop, the gathered version the engine uses, and
     Method 2 torch's scaled_dot_product_attention on a contiguous copy; the maximum differences.
  4. The whole model (§4.3): our OPT on the paged cache against transformers' OPT on its own cache: logits of
     the prompt step and of 5 decoding steps, then 8 Alpaca prompts served together, greedy, token for token.
  5. Parallel sampling and copy-on-write (§4.4, Figure 8): two samples share the prompt's blocks; reference
     counts, the copy, and a check of every sampled token's log-probability against the library.
  6. Scheduling and preemption by recomputation (§4.5): 6 requests in a 12-block cache; the events, the outputs
     against the library, and the time of recomputing a cache in one pass against writing it token by token.
  7. Serving (§4, §6 in small): 64 Alpaca requests through a 2,048-slot KV cache, paged against one contiguous
     chunk per request booked for 512 tokens. Throughput, batch size and memory use.
  8. Sharing in numbers (§6.3 in small): 16 Alpaca prompts with 4 samples each: blocks held with sharing
     against blocks without sharing.
Model: facebook/opt-125m, FP32 on the CPU. Timing is the fastest of a few runs on a laptop that runs other work.
Run: python vllm_part6.py   (about 5 minutes on an M-series Mac)"""
import os, statistics, time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save, HERE
from minivllm import (LLMEngine, SamplingParams, paged_attention, paged_attention_eq4)
from minivllm.block_manager import cdiv

log = Log('part6')
torch.manual_seed(0)
OUT = {}
tok = AutoTokenizer.from_pretrained('facebook/opt-125m')
hf = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', dtype=torch.float32).eval()
EOS = tok.eos_token_id


def words(ids):
    return [tok.decode([t]).strip() or repr(tok.decode([t])) for t in ids]


@torch.no_grad()
def library_greedy(ids, n):
    """Method 2: the library's model with its own contiguous KV cache, one greedy token at a time."""
    out = hf(torch.tensor([ids]), use_cache=True)
    toks, logits = [], [out.logits[0, -1]]
    for _ in range(n):
        t = int(out.logits[0, -1].argmax())
        toks.append(t)
        out = hf(torch.tensor([[t]]), past_key_values=out.past_key_values, use_cache=True)
        logits.append(out.logits[0, -1])
    return toks, logits


alp = load_dataset('tatsu-lab/alpaca')['train']


def alpaca(i):
    r = alp[i]
    prompt = r['instruction'] + ('\n' + r['input'] if r['input'] else '')
    return tok(prompt).input_ids, len(tok(r['output']).input_ids) - 1      # minus the start token


# ---------------------------------------------------------------- 1. the package
log('== 1. The package minivllm')
PKG = os.path.join(HERE, 'minivllm')
files = ['block_manager.py', 'kv_cache.py', 'attention.py', 'model.py', 'sequence.py', 'scheduler.py', 'engine.py']
counts = {f: sum(1 for l in open(os.path.join(PKG, f)) if l.strip()) for f in files}
for f in files:
    log(f'  {f:<17} {counts[f]:>4} lines')
log(f'  total             {sum(counts.values()):>4} lines of Python (no CUDA)')
OUT['package'] = counts

# ---------------------------------------------------------------- 2. block tables, Figures 6 and 7
log('== 2. Block tables (Figures 6 and 7): block size 4, a cache of 9 physical blocks')
eng = LLMEngine(hf, num_blocks=9, block_size=4)
A = eng.add_request(tok('Four score and seven years ago').input_ids, SamplingParams(max_tokens=4))
B = eng.add_request(tok('It was the best of times').input_ids, SamplingParams(max_tokens=4))
trace = []


def show(g, name):
    s = g.seqs[0]
    t = eng.bm.tables[s.seq_id]
    toks = words(s.token_ids[:t.num_tokens])
    rows = []
    for j, b in enumerate(t.blocks):
        cell = toks[j * 4:(j + 1) * 4]
        rows.append({'logical': j, 'physical': b, 'filled': t.filled(j, 4), 'tokens': cell})
        log(f'    {name} logical {j} -> physical {b}, filled {t.filled(j, 4)}: {" | ".join(cell)}')
    return rows


for step, label in enumerate(['prompt step', 'decoding step 1', 'decoding step 2']):
    eng.step()
    log(f'  after the {label}:')
    trace.append({'label': label, 'A': show(A, 'A'), 'B': show(B, 'B'), 'free': eng.bm.allocator.num_free})
    log(f'    free blocks: {eng.bm.allocator.num_free} of 9')
OUT['fig6'] = trace

# ---------------------------------------------------------------- 3. PagedAttention, Eq. 4
log('== 3. PagedAttention (Eq. 4): one query against 10 tokens in 3 scattered blocks of 4 (12 heads of 64)')
Bs, H, D = 4, 12, 64
k_blocks, v_blocks = torch.randn(9, Bs, H, D), torch.randn(9, Bs, H, D)
table, n = torch.tensor([7, 1, 3]), 10                                  # the block table of Figure 5's sentence
q = torch.randn(H, D)
o_eq4 = paged_attention_eq4(q, k_blocks, v_blocks, table, n)            # Method 1: block by block
o_gather = paged_attention(q[None], k_blocks, v_blocks, table, torch.tensor([n - 1]))[0]
K = torch.cat([k_blocks[b] for b in table])[:n]                         # Method 2: copy into one contiguous tensor
V = torch.cat([v_blocks[b] for b in table])[:n]
o_lib = F.scaled_dot_product_attention(q[:, None, :], K.transpose(0, 1), V.transpose(0, 1))[:, 0]
d1, d2 = (o_eq4 - o_lib).abs().max().item(), (o_gather - o_lib).abs().max().item()
log(f'  Eq. 4 block by block vs torch on a contiguous copy: max difference {d1:.1e}')
log(f'  gathered version (used by the engine) vs torch:     max difference {d2:.1e}')
k2, v2 = k_blocks.clone(), v_blocks.clone()
k2[[0, 5, 8]], v2[[0, 5, 8]] = k_blocks[table], v_blocks[table]         # move the 3 blocks elsewhere in memory
o_moved = paged_attention_eq4(q, k2, v2, torch.tensor([0, 5, 8]), n)
log(f'  same blocks moved to physical 0, 5, 8 (new block table): max difference {(o_moved - o_eq4).abs().max().item():.1e}')
OUT['eq4'] = {'eq4_vs_lib': d1, 'gather_vs_lib': d2}

# ---------------------------------------------------------------- 4. the whole model against the library
log('== 4. Our OPT on the paged cache against transformers (FP32, CPU)')
ids = tok('Four score and seven years ago our fathers').input_ids
eng = LLMEngine(hf, num_blocks=16, block_size=4)
g = eng.add_request(ids, SamplingParams(max_tokens=6, ignore_eos=True))
ref_toks, ref_logits = library_greedy(ids, 6)
diffs = []
for i in range(6):
    eng.step()
    diffs.append((eng.last_logits[g.seqs[0].seq_id] - ref_logits[i]).abs().max().item())
log(f'  prompt of {len(ids)} tokens, then 5 decoding steps: max logit difference per step '
    + ', '.join(f'{d:.1e}' for d in diffs))
log(f'  greedy tokens: ours {words(g.seqs[0].output_ids)}; library {words(ref_toks)}')
OUT['logits'] = diffs

prompts = [alpaca(i)[0] for i in range(8)]
eng = LLMEngine(hf, num_blocks=128, block_size=16)
groups = [eng.add_request(p, SamplingParams(max_tokens=48, ignore_eos=True)) for p in prompts]
eng.run()
same = tok_same = 0
for p, g in zip(prompts, groups):
    ref, _ = library_greedy(p, 48)
    ours = g.seqs[0].output_ids
    same += ours == ref
    tok_same += sum(a == b for a, b in zip(ours, ref))
log(f'  8 Alpaca prompts ({min(map(len, prompts))} to {max(map(len, prompts))} tokens) served together, 48 greedy tokens each:')
log(f'  identical outputs {same} of 8; identical tokens {tok_same} of {8 * 48}; steps {eng.step_no}')
log(f'  first prompt: {tok.decode(prompts[0][1:])!r}')
log(f'  our answer:   {tok.decode(groups[0].seqs[0].output_ids[:24])!r} ...')
OUT['greedy'] = {'same': same, 'tokens_same': tok_same, 'steps': eng.step_no}

# ---------------------------------------------------------------- 5. parallel sampling and copy-on-write
log('== 5. Parallel sampling and copy-on-write (Figure 8): 2 samples, block size 4')
eng = LLMEngine(hf, num_blocks=9, block_size=4)
g = eng.add_request(tok('Four score and seven years ago').input_ids,
                    SamplingParams(n=2, max_tokens=12, temperature=1.0, ignore_eos=True, seed=3))
cow = []
for step in range(3):
    out = eng.step()
    a1, a2 = g.seqs
    t1, t2 = eng.bm.tables[a1.seq_id], eng.bm.tables[a2.seq_id]
    rc = {b: eng.bm.allocator.ref_count[b] for b in sorted(set(t1.blocks + t2.blocks))}
    log(f'  step {step}: A1 blocks {t1.blocks}, A2 blocks {t2.blocks}; ref counts {rc}; copies {out.copies}')
    cow.append({'A1': list(t1.blocks), 'A2': list(t2.blocks), 'ref': {str(k): v for k, v in rc.items()},
                'copies': out.copies, 'A1_tokens': words(a1.token_ids[:t1.num_tokens]),
                'A2_tokens': words(a2.token_ids[:t2.num_tokens])})
eng.run()
worst = 0.0
with torch.no_grad():
    for s in g.seqs:
        lp = torch.log_softmax(hf(torch.tensor([s.token_ids])).logits[0], -1)
        lib = [lp[s.prompt_len - 1 + i, t].item() for i, t in enumerate(s.output_ids)]
        worst = max(worst, max(abs(a - b) for a, b in zip(lib, s.logprobs)))
for i, s in enumerate(g.seqs):
    log(f'  sample A{i + 1}: {tok.decode(s.output_ids)!r}')
log(f'  log-probability of all {sum(len(s.output_ids) for s in g.seqs)} sampled tokens, ours vs the library on the full text: '
    f'max difference {worst:.1e}')
OUT['cow'] = {'trace': cow, 'logprob_diff': worst}

# ---------------------------------------------------------------- 6. scheduling and preemption
log('== 6. Scheduling and preemption by recomputation: 6 requests, 64 greedy tokens each, a cache of 12 blocks of 16')
reqs = [alpaca(i)[0] for i in range(8, 14)]
eng = LLMEngine(hf, num_blocks=12, block_size=16)
groups = [eng.add_request(p, SamplingParams(max_tokens=64, ignore_eos=True)) for p in reqs]
events, timeline = [], []
while eng.scheduler.has_work():
    out = eng.step()
    s = eng.step_no - 1
    for gp in out.preempted:
        events.append(f'step {s:>3}: preempt request {gp.request_id} (it has {len(gp.seqs[0].output_ids)} new tokens); '
                      f'free blocks now {eng.bm.allocator.num_free}')
    for seq, slots in out.prefill:
        if len(slots) > seq.prompt_len:                     # more than the prompt: a recomputation
            events.append(f'step {s:>3}: request {seq.group.request_id} back: recompute {len(slots)} tokens '
                          f'({seq.prompt_len} prompt + {len(slots) - seq.prompt_len} written) in one pass')
    timeline.append({'running': [gp.request_id for gp in eng.scheduler.running],
                     'waiting': [gp.request_id for gp in eng.scheduler.waiting], 'free': eng.bm.allocator.num_free})
for e in events[:10]:
    log('  ' + e)
if len(events) > 10:
    log(f'  ... {len(events) - 10} more events')
ok = sum(g.seqs[0].output_ids == library_greedy(p, 64)[0] for p, g in zip(reqs, groups))
log(f'  prompts {[len(p) for p in reqs]} tokens; {eng.step_no} steps; {eng.scheduler.num_preemptions} preemptions')
log(f'  outputs identical to the library (which never preempts): {ok} of 6')
OUT['preempt'] = {'events': events, 'timeline': timeline, 'same': ok, 'preemptions': eng.scheduler.num_preemptions,
                  'steps': eng.step_no, 'prompt_lens': [len(p) for p in reqs]}


def fastest(fn, repeats=3):
    best = 1e9
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best


def recompute():
    e = LLMEngine(hf, num_blocks=16, block_size=16)
    e.add_request(alpaca(20)[0][:16] + list(range(1000, 1112)), SamplingParams(max_tokens=1))
    e.run()


def token_by_token():
    e = LLMEngine(hf, num_blocks=16, block_size=16)
    e.add_request(alpaca(20)[0][:16], SamplingParams(max_tokens=113, ignore_eos=True))
    e.run()


t_re, t_tok = fastest(recompute), fastest(token_by_token)
log(f'  the cache of 128 tokens: one pass {t_re * 1000:.0f} ms; a 16-token prompt then 112 tokens one by one '
    f'{t_tok * 1000:.0f} ms ({t_tok / t_re:.0f}x longer)')
OUT['recompute'] = {'one_pass_ms': t_re * 1000, 'token_by_token_ms': t_tok * 1000}

# ---------------------------------------------------------------- 7. serving: paged against contiguous
N_REQ, BLOCKS, BS, MAXLEN = 64, 128, 16, 512
work = [alpaca(i) for i in range(100, 100 + N_REQ)]
work = [(p, min(max(o, 1), 128)) for p, o in work]
total_out = sum(o for _, o in work)
log(f'== 7. Serving {N_REQ} Alpaca requests: KV cache of {BLOCKS} blocks x {BS} = {BLOCKS * BS} token slots')
log(f'  prompts: mean {statistics.mean(len(p) for p, _ in work):.1f} tokens; outputs: the length of the dataset answer, '
    f'at most 128, mean {statistics.mean(o for _, o in work):.1f}; {total_out} tokens to write')
serve = {}
for name in ('contiguous', 'paged'):
    runs = []
    for rep in range(2):
        e = LLMEngine(hf, num_blocks=BLOCKS, block_size=BS, manager='paged' if name == 'paged' else 'contiguous',
                      max_len=MAXLEN)
        for p, o in work:
            e.add_request(p, SamplingParams(max_tokens=o, ignore_eos=True))
        runs.append((e.run(), e))
    t, e = min(runs, key=lambda r: r[0])
    st = e.stats
    used = sum(x['tokens'] for x in st) / sum(x['blocks_held'] * BS for x in st)
    serve[name] = {'seconds': t, 'all_seconds': [r[0] for r in runs], 'steps': len(st),
                   'mean_batch': statistics.mean(x['running'] for x in st), 'max_batch': max(x['running'] for x in st),
                   'used_share': used, 'preemptions': e.scheduler.num_preemptions,
                   'tok_s': total_out / t, 'req_s': N_REQ / t,
                   'curve': [x['running'] for x in st], 'held': [x['blocks_held'] for x in st],
                   'tokens': [x['tokens'] for x in st]}
    bytes_block = e.cache.bytes_per_block()
log(f'  {"":<11}{"seconds":>9}{"tokens/s":>10}{"requests/s":>12}{"steps":>7}{"mean batch":>12}{"max batch":>11}'
    f'{"slots holding a token":>23}{"preemptions":>13}')
for name, r in serve.items():
    log(f'  {name:<11}{r["seconds"]:>9.1f}{r["tok_s"]:>10.0f}{r["req_s"]:>12.2f}{r["steps"]:>7}{r["mean_batch"]:>12.1f}'
        f'{r["max_batch"]:>11}{r["used_share"] * 100:>22.1f}%{r["preemptions"]:>13}')
c, p = serve['contiguous'], serve['paged']
log(f'  runs (seconds): contiguous {", ".join(f"{x:.1f}" for x in c["all_seconds"])}; paged {", ".join(f"{x:.1f}" for x in p["all_seconds"])}')
log(f'  paged / contiguous: throughput x{p["tok_s"] / c["tok_s"]:.1f}, mean batch x{p["mean_batch"] / c["mean_batch"]:.1f}')
kv_mb = BLOCKS * bytes_block / 1e6
need = p['mean_batch'] * MAXLEN / BS * bytes_block / 1e6
log(f'  the KV cache is {kv_mb:.0f} MB in both runs ({bytes_block / BS / 1024:.0f} KB per token in FP32); '
    f'booking 512 slots each, the paged mean batch would need {need:.0f} MB')
OUT['serve'] = serve
OUT['serve_setup'] = {'requests': N_REQ, 'blocks': BLOCKS, 'block_size': BS, 'max_len': MAXLEN, 'kv_mb': kv_mb,
                      'need_mb': need, 'tokens': total_out}

# ---------------------------------------------------------------- 8. sharing in numbers
log('== 8. Parallel sampling, 4 samples per prompt: blocks held with sharing against without (16 Alpaca prompts)')
e = LLMEngine(hf, num_blocks=512, block_size=16)
for i in range(16):
    p, o = alpaca(200 + i)
    e.add_request(p, SamplingParams(n=4, max_tokens=min(max(o, 1), 64), temperature=1.0, ignore_eos=True, seed=i))
e.run()
st = [x for x in e.stats if x['blocks_no_sharing']]
saving = statistics.mean(1 - x['blocks_held'] / x['blocks_no_sharing'] for x in st)
copies = sum(x['copies'] for x in e.stats)
log(f'  {len(st)} steps; mean blocks held {statistics.mean(x["blocks_held"] for x in st):.1f}, without sharing '
    f'{statistics.mean(x["blocks_no_sharing"] for x in st):.1f}; mean saving {saving * 100:.1f}%; copy-on-write copies {copies}')
OUT['sharing'] = {'saving': saving, 'copies': copies, 'steps': len(st),
                  'held': [x['blocks_held'] for x in st], 'no_sharing': [x['blocks_no_sharing'] for x in st]}
save('part6', OUT)
