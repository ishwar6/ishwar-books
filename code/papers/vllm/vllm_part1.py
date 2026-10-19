"""Part 1 of vLLM, explained: the big idea, run on a laptop.

What it shows (every number is printed and saved to results/part1_stdout.txt and results/part1.json):
  1. The size of the KV cache of one token: the paper's formula on the real OPT-13B config (Method 1), and the bytes
     of the real cache tensors of OPT-125M after a prompt (Method 2, checked against the same formula).
  2. Figure 1 (left) as arithmetic: 13B parameters in FP16 on a 40 GB A100, and how many tokens fit in the rest.
  3. Why the cache exists: greedy generation that recomputes every token (no cache) against our own loop with a
     KV cache, against model.generate (the library). All three give the same tokens; the cache is faster.
  4. The two phases: the prompt phase in one forward pass against the generation phase one token at a time.
  5. Batching: one decode step for 1 to 64 requests at once. The step time grows slowly; the throughput grows fast.
  6. Figure 2 in miniature: the KV cache memory that holds real tokens when each request reserves 2048 slots,
     the next power of two, its exact length, or 16-token blocks on demand. Lengths of real Alpaca requests
     (tatsu-lab/alpaca, OPT tokenizer). Counted step by step (Method 1) and with a closed form (Method 2).
  7. The paper's headline numbers, checked with arithmetic.
Model: facebook/opt-125m (the smallest model of the OPT family that the paper serves), FP16 on Apple MPS.
Run: python vllm_part1.py   (about 2 minutes on an M-series Mac)"""
import json, math, random
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save, timer

log = Log('part1')
SMALL = 'facebook/opt-125m'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(0)


def sync():
    if DEV == 'mps':
        torch.mps.synchronize()


def fastest(fn, repeats=5):
    """Run fn several times; return the fastest wall time in seconds (other work on the laptop only adds time)."""
    best = 1e9
    for _ in range(repeats):
        sync(); clock = timer(); fn(); sync()
        best = min(best, clock())
    return best


# ---------------------------------------------------------------- 1. the KV cache of one token
log('== 1. The KV cache of one token')


def kv_bytes_per_token(cfg, bytes_per_value=2):
    """The paper's formula (§3): 2 (key and value) x hidden size x number of layers x bytes per FP16 number."""
    return 2 * cfg['hidden_size'] * cfg['num_hidden_layers'] * bytes_per_value


big = json.load(open(hf_hub_download('facebook/opt-13b', 'config.json')))
per_tok_13b = kv_bytes_per_token(big)
log(f"OPT-13B config: {big['num_hidden_layers']} layers, hidden size {big['hidden_size']}, "
    f"max length {big['max_position_embeddings']} tokens")
log(f'Method 1, formula: 2 x {big["hidden_size"]} x {big["num_hidden_layers"]} x 2 bytes = {per_tok_13b:,} bytes '
    f'= {per_tok_13b / 1024:.0f} KB per token')
per_req_13b = per_tok_13b * big['max_position_embeddings']
log(f"one request of {big['max_position_embeddings']} tokens: {per_req_13b:,} bytes = {per_req_13b / 1e9:.2f} GB")

tok = AutoTokenizer.from_pretrained(SMALL)
model = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float16).to(DEV).eval()
small = model.config.to_dict()
PROMPT = 'Four score and seven years ago our fathers brought forth on this continent a new nation'
ids = tok(PROMPT, return_tensors='pt').input_ids.to(DEV)
with torch.no_grad():
    cache = model(ids, use_cache=True).past_key_values


def cache_bytes(cache):
    """Method 2: add up the bytes of every key and value tensor that the library really keeps."""
    return sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))


n = ids.shape[1]
k0 = cache.layers[0].keys
log(f'OPT-125M: {small["num_hidden_layers"]} layers, hidden size {small["hidden_size"]}; prompt of {n} tokens')
log(f'shape of the keys of layer 0: {tuple(k0.shape)}  (batch, heads, tokens, head size), {k0.dtype}')
log(f'Method 1, formula: {kv_bytes_per_token(small):,} bytes per token x {n} tokens = {kv_bytes_per_token(small) * n:,} bytes')
log(f'Method 2, real tensors: {cache_bytes(cache):,} bytes; agree: {cache_bytes(cache) == kv_bytes_per_token(small) * n}')

# ---------------------------------------------------------------- 2. Figure 1 (left) as arithmetic
log('')
log('== 2. Figure 1 (left): a 13B model on a 40 GB A100')
GPU = 40e9
params = 13e9
w_bytes = params * 2
kv_room = 0.30 * GPU
log(f'weights: 13B parameters x 2 bytes = {w_bytes / 1e9:.0f} GB = {100 * w_bytes / GPU:.0f}% of 40 GB')
log(f'KV cache room at 30% of the GPU: {kv_room / 1e9:.0f} GB = {kv_room / per_tok_13b:,.0f} tokens of OPT-13B')
log(f'requests of 2048 tokens that fit: {kv_room / per_req_13b:.1f}')

# ---------------------------------------------------------------- 3. why the cache exists, three ways
log('')
log('== 3. Greedy generation three ways: no cache, our own KV cache loop, model.generate')
N_NEW = 64


@torch.no_grad()
def generate_no_cache(ids, n_new):
    """Recompute the keys and values of every earlier token at every step."""
    for _ in range(n_new):
        logits = model(ids, use_cache=False).logits
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], dim=1)
    return ids


@torch.no_grad()
def generate_with_cache(ids, n_new):
    """The prompt phase once, then one new token per step; the cache keeps every earlier key and value."""
    out = model(ids, use_cache=True)                       # prompt phase: all prompt tokens in one pass
    cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    toks = [nxt]
    for _ in range(n_new - 1):                             # generation phase: one token in, one token out
        out = model(nxt, past_key_values=cache, use_cache=True)
        cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        toks.append(nxt)
    return torch.cat([ids] + toks, dim=1), cache


a = generate_no_cache(ids, N_NEW)
b, cache_b = generate_with_cache(ids, N_NEW)
with torch.no_grad():
    c = model.generate(ids, max_new_tokens=N_NEW, do_sample=False, pad_token_id=tok.pad_token_id)
log(f'{N_NEW} new tokens; same tokens, no cache vs our cache: {torch.equal(a, b)}; our cache vs generate: {torch.equal(b, c)}')
log('text: ' + repr(tok.decode(b[0, n:], skip_special_tokens=True)[:90]) + ' ...')
log(f'cache after generation: {cache_b.get_seq_length()} tokens = {cache_bytes(cache_b):,} bytes '
    f'(the last token gets its key and value at the next step)')
t_nc = fastest(lambda: generate_no_cache(ids, N_NEW), 3)
t_c = fastest(lambda: generate_with_cache(ids, N_NEW), 3)
log(f'time: no cache {t_nc:.3f} s, with cache {t_c:.3f} s ({t_nc / t_c:.1f}x)')
long_ids = tok(' '.join(['The history of the city is long.'] * 60), return_tensors='pt').input_ids[:, :400].to(DEV)
t_nc_l = fastest(lambda: generate_no_cache(long_ids, N_NEW), 3)
t_c_l = fastest(lambda: generate_with_cache(long_ids, N_NEW), 3)
log(f'with a 400-token prompt: no cache {t_nc_l:.3f} s, with cache {t_c_l:.3f} s ({t_nc_l / t_c_l:.1f}x)')

# ---------------------------------------------------------------- 4. the two phases
log('')
log('== 4. The prompt phase against the generation phase (OPT-125M, 400 tokens)')
P = long_ids.shape[1]
t_prompt = fastest(lambda: model(long_ids, use_cache=True))
out = model(long_ids[:, :1], use_cache=True)


@torch.no_grad()
def one_by_one(ids):
    """Feed the same 400 tokens one at a time, as the generation phase must."""
    cache = None
    for i in range(ids.shape[1]):
        cache = model(ids[:, i:i + 1], past_key_values=cache, use_cache=True).past_key_values


with torch.no_grad():
    t_seq = fastest(lambda: one_by_one(long_ids), 3)
log(f'prompt phase, {P} tokens in one pass:   {t_prompt * 1000:7.1f} ms = {P / t_prompt:9,.0f} tokens/s')
log(f'the same {P} tokens one at a time:      {t_seq * 1000:7.1f} ms = {P / t_seq:9,.0f} tokens/s')
log(f'one pass is {t_seq / t_prompt:.0f}x faster for the same work')

# ---------------------------------------------------------------- 5. batching the generation phase
log('')
log('== 5. One decode step for a batch of requests (each with a 128-token context)')
CTX = 128
BATCH = {}
log(f'{"batch":>5} {"step ms":>8} {"tokens/s":>9}')
for bs in [1, 2, 4, 8, 16, 32, 64]:
    ctx = torch.randint(100, 20000, (bs, CTX), device=DEV)
    with torch.no_grad():
        cache = model(ctx, use_cache=True).past_key_values
    step_in = torch.randint(100, 20000, (bs, 1), device=DEV)

    def step():
        with torch.no_grad():
            model(step_in, past_key_values=cache, use_cache=True)
            for layer in cache.layers:                      # drop the token just added, so every run is the same
                layer.keys, layer.values = layer.keys[:, :, :CTX], layer.values[:, :, :CTX]
    t = fastest(step, 7)
    BATCH[bs] = {'step_ms': t * 1000, 'tokens_per_s': bs / t}
    log(f'{bs:>5} {t * 1000:>8.2f} {bs / t:>9,.0f}')
log(f'from batch 1 to 64: step time x{BATCH[64]["step_ms"] / BATCH[1]["step_ms"]:.1f}, '
    f'throughput x{BATCH[64]["tokens_per_s"] / BATCH[1]["tokens_per_s"]:.1f}')

# ---------------------------------------------------------------- 6. Figure 2 in miniature
log('')
log('== 6. How much reserved KV cache memory holds real tokens (Alpaca lengths, OPT tokenizer)')
alp = load_dataset('tatsu-lab/alpaca')['train']
MAXLEN, BLOCK = 2048, 16
rng = random.Random(0)
sample = [alp[i] for i in rng.sample(range(len(alp)), 2000)]
REQS = []
for r in sample:
    prompt = r['instruction'] + ('\n' + r['input'] if r['input'] else '')
    p_len, o_len = len(tok(prompt).input_ids), len(tok(r['output']).input_ids)
    if o_len > 0 and p_len + o_len <= MAXLEN:
        REQS.append((p_len, o_len))
p_mean = sum(p for p, _ in REQS) / len(REQS)
o_mean = sum(o for _, o in REQS) / len(REQS)
log(f'{len(REQS)} requests; mean prompt {p_mean:.1f} tokens, mean output {o_mean:.1f} tokens, '
    f'longest {max(p + o for p, o in REQS)} tokens')

SCHEMES = {                                                # slots held, given current length, prompt and final length
    'max (2048)': lambda cur, p, final: MAXLEN,
    'pow2': lambda cur, p, final: p + 2 ** math.ceil(math.log2(final - p)),   # output booked up to 2x (25 -> 32)
    'oracle': lambda cur, p, final: final,
    f'paged ({BLOCK})': lambda cur, p, final: BLOCK * math.ceil(cur / BLOCK),
}


def simulate(alloc):
    """Method 1: walk every request step by step. At each step, count the slots that hold a token (used),
    the slots kept for tokens that will come later (reserved) and the slots that no token will ever fill (waste)."""
    used = reserved = waste = 0
    for p, o in REQS:
        final = p + o
        for cur in range(p, final + 1):                    # after the prompt phase, then after each new token
            a = alloc(cur, p, final)
            used += cur
            fut = min(a, final) - cur                       # slots that a later token of this request will fill
            reserved += fut
            waste += a - cur - fut
    return used, reserved, waste


def closed_form(name):
    """Method 2: the same sums without the loop. Over the steps cur = p..p+o, the sum of cur is
    (o + 1)(2p + o) / 2. Pre-allocated schemes give a fixed a per request; paged sums its blocks."""
    used = reserved = waste = 0
    for p, o in REQS:
        final, steps = p + o, o + 1
        s_cur = steps * (2 * p + o) // 2
        used += s_cur
        if name.startswith('paged'):                       # blocks grow with the request: sum them per step
            blocks = [BLOCK * math.ceil(c / BLOCK) for c in range(p, final + 1)]
            reserved += sum(min(a, final) for a in blocks) - s_cur
            waste += sum(a - min(a, final) for a in blocks)
        else:
            a = SCHEMES[name](None, p, final)
            reserved += steps * final - s_cur
            waste += steps * (a - final)
    return used, reserved, waste


FIG2 = {}
log(f'{"scheme":<12} {"token states":>12} {"reserved":>9} {"never used":>11}  {"Method 2 agrees":>15}')
for name, alloc in SCHEMES.items():
    u, r, w = simulate(alloc)
    total = u + r + w
    FIG2[name] = {'tokens': 100 * u / total, 'reserved': 100 * r / total, 'waste': 100 * w / total,
                  'mean_slots': total / sum(o + 1 for _, o in REQS)}
    log(f'{name:<12} {FIG2[name]["tokens"]:>11.1f}% {FIG2[name]["reserved"]:>8.1f}% {FIG2[name]["waste"]:>10.1f}%  '
        f'{str(closed_form(name) == (u, r, w)):>15}')
log('external fragmentation is not simulated: we place no requests in a real address space')
log('requests of OPT-13B that fit in 12 GB of KV cache, on average over their lifetime:')
for name, f in FIG2.items():
    fit = kv_room / per_tok_13b / f['mean_slots']
    FIG2[name]['fit_12gb'] = fit
    log(f'  {name:<12} {f["mean_slots"]:7.1f} slots per request -> {fit:6.1f} requests')

# ---------------------------------------------------------------- 7. the paper's numbers, checked
log('')
log("== 7. The paper's headline numbers, checked")
fig2_paper = {'Orca (Max)': 20.4, 'Orca (Pow2)': 26.8, 'Orca (Oracle)': 38.2, 'vLLM': 96.3}
for k, v in fig2_paper.items():
    log(f'Figure 2, token states in {k:<14} {v:5.1f}%   -> waste {100 - v:5.1f}%')
log(f'Figure 2: vLLM keeps {96.3 / 20.4:.1f}x to {96.3 / 38.2:.1f}x more of the memory for real tokens')
log(f'§3: 819,200 bytes x 2048 tokens = {819200 * 2048 / 2**30:.2f} GiB = {819200 * 2048 / 1e9:.2f} GB ("1.6 GB")')

save('part1', {
    'opt13b': {'layers': big['num_hidden_layers'], 'hidden': big['hidden_size'], 'bytes_per_token': per_tok_13b,
               'bytes_per_request': per_req_13b},
    'opt125m': {'layers': small['num_hidden_layers'], 'hidden': small['hidden_size'], 'prompt_tokens': n,
                'cache_bytes': cache_bytes(cache), 'formula_bytes': kv_bytes_per_token(small) * n},
    'fig1': {'weights_gb': w_bytes / 1e9, 'kv_tokens_12gb': kv_room / per_tok_13b, 'max_requests': kv_room / per_req_13b},
    'gen': {'n_new': N_NEW, 'no_cache_s': t_nc, 'cache_s': t_c, 'no_cache_long_s': t_nc_l, 'cache_long_s': t_c_l},
    'phases': {'tokens': P, 'prompt_ms': t_prompt * 1000, 'one_by_one_ms': t_seq * 1000},
    'batch': BATCH, 'fig2': FIG2, 'n_requests': len(REQS), 'prompt_mean': p_mean, 'output_mean': o_mean,
    'device': DEV,
})
