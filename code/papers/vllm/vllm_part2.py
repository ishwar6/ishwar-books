"""Part 2 of vLLM, explained: why memory is the bottleneck (§2, §3, Figure 3), run on a laptop.

What it shows (every number is printed and saved to results/part2_stdout.txt and results/part2.json):
  1. Eq. 1, the chain rule: the probability of a sentence as a product of next-token probabilities. Method 1 runs
     the model once per prefix; Method 2 reads all the conditionals from one pass (the prompt phase).
  2. The KV cache of a token depends on its position and on the tokens before it: the same word "the" at two
     positions, and the same word at the same position after two different openings, in OPT-125M.
  3. Matrix-vector against matrix-matrix: one 5120 x 5120 weight matrix (the size of OPT-13B) times 1 to 512
     token vectors on the laptop GPU. The time hardly changes until the batch is large.
  4. Batching (§2.3): request-level batching with padding against iteration-level batching, simulated step by step
     on the output lengths of real Alpaca requests (Method 1), padding checked with a closed form (Method 2).
  5. Large KV cache (§3): KV cache bytes per token and per full request from the real configs of four OPT models.
  6. Complex decoding (§3): parallel sampling stores the same prompt cache once per sample; beam search keeps
     beams that share a long prefix (model.generate on OPT-125M).
  7. Unknown output lengths (§3): 20 sampled answers to one prompt, and the spread of Alpaca output lengths.
  8. Figure 3: the slots of requests A (2048) and B (512) counted as token states, reserved and never used.
  9. Compaction: how fast this laptop copies GPU memory, and what moving 12 GB of KV cache would cost.
Model: facebook/opt-125m, FP32 on the CPU for exact checks (1, 2, 6, 7) and FP16 on Apple MPS for timing (3, 9).
Run: python vllm_part2.py   (about 2 minutes on an M-series Mac)"""
import json, math, random, statistics
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save, timer

log = Log('part2')
SMALL = 'facebook/opt-125m'
GPU = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(0)
OUT = {}

tok = AutoTokenizer.from_pretrained(SMALL)
model = AutoModelForCausalLM.from_pretrained(SMALL, dtype=torch.float32).eval()


def sync():
    if GPU == 'mps':
        torch.mps.synchronize()


def fastest(fn, repeats=7):
    """Run fn several times; return the fastest wall time in seconds (other work on the laptop only adds time)."""
    best = 1e9
    for _ in range(repeats):
        sync(); clock = timer(); fn(); sync()
        best = min(best, clock())
    return best


# ---------------------------------------------------------------- 1. Eq. 1, the chain rule
log('== 1. Eq. 1: the probability of a sentence is a product of next-token probabilities')
SENT = 'Four score and seven years ago our fathers'
ids = tok(SENT, return_tensors='pt').input_ids             # OPT puts its start token </s> first
words = [tok.decode(t) for t in ids[0, 1:]]


@torch.no_grad()
def chain_rule_one_by_one(ids):
    """Method 1: for each position i, run the model on x_1..x_{i-1} and read P(x_i | x_1..x_{i-1})."""
    probs = []
    for i in range(1, ids.shape[1]):
        logits = model(ids[:, :i]).logits[0, -1]           # the model sees only the tokens before position i
        probs.append(torch.softmax(logits, -1)[ids[0, i]].item())
    return probs


@torch.no_grad()
def chain_rule_one_pass(ids):
    """Method 2: one pass over the whole sentence. Thanks to the causal mask, row i of the output already
    holds P(x_{i+1} | x_1..x_i), so every conditional comes out at once (this is the prompt phase)."""
    logits = model(ids).logits[0, :-1]                      # (n - 1, vocabulary)
    return torch.softmax(logits, -1).gather(1, ids[0, 1:, None])[:, 0].tolist()


p1, p2 = chain_rule_one_by_one(ids), chain_rule_one_pass(ids)
log(f'{"token":<10} {"P(token | earlier)":>18}')
for w, p in zip(words, p1):
    log(f'{w!r:<10} {p:>18.2e}')
prod1, prod2 = math.prod(p1), math.prod(p2)
with torch.no_grad():
    loss = model(ids, labels=ids).loss.item()               # the library: mean of -log P over the same tokens
prod3 = math.exp(-loss * len(p1))
log(f'product, Method 1 ({len(p1)} passes): {prod1:.3e}')
log(f'product, Method 2 (1 pass):     {prod2:.3e}')
log(f'library loss {loss:.4f} -> exp(-loss x {len(p1)}) = {prod3:.3e}; max difference of the conditionals '
    f'{max(abs(a - b) for a, b in zip(p1, p2)):.1e}')
OUT['chain'] = {'words': words, 'probs': p1, 'product': prod1, 'loss': loss}

# ---------------------------------------------------------------- 2. the KV cache depends on position and context
log('')
log('== 2. The same token gets a different KV cache at a different position or after different words')


@torch.no_grad()
def keys_of(text):
    """The keys of every token at every layer: a list over layers of (tokens, hidden) tensors."""
    cache = model(tok(text, return_tensors='pt').input_ids, use_cache=True).past_key_values
    return [layer.keys[0].transpose(0, 1).reshape(layer.keys.shape[2], -1) for layer in cache.layers]


def cos(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


T1 = 'the cat sat on the mat'
t1 = tok(T1).input_ids
the = [i for i, t in enumerate(t1) if tok.decode(t).strip() == 'the']
k = keys_of(T1)
log(f'{T1!r}: "the" at token positions {the[0]} and {the[1]}')
OUT['positions'] = {}
for layer in (0, 11):
    a, b = k[layer][the[0]], k[layer][the[1]]
    log(f'  layer {layer:>2}: keys of the two "the" are equal: {torch.equal(a, b)}; cosine similarity {cos(a, b):.3f}')
    OUT['positions'][layer] = cos(a, b)

T2, T3 = 'I saw the', 'We ate the'
k2, k3 = keys_of(T2), keys_of(T3)
log(f'{T2!r} against {T3!r}: "the" is token 3 in both')
OUT['context'] = {}
for layer in (0, 1, 11):
    a, b = k2[layer][3], k3[layer][3]
    log(f'  layer {layer:>2}: max difference of the keys of "the" {(a - b).abs().max().item():.2e}; '
        f'cosine similarity {cos(a, b):.3f}')
    OUT['context'][layer] = (a - b).abs().max().item()
log('layer 0 sees only the token and its position; from layer 1 on, the key carries the earlier words')

# ---------------------------------------------------------------- 3. matrix-vector against matrix-matrix
log('')
log('== 3. One 5120 x 5120 FP16 weight matrix (OPT-13B size) times n token vectors, on the laptop GPU')
D = 5120
W = torch.randn(D, D, dtype=torch.float16, device=GPU)
log(f'{"n tokens":>8} {"time ms":>8} {"TFLOP/s":>8} {"FLOPs per byte":>15}')
OUT['matmul'] = {}
for n in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
    X = torch.randn(D, n, dtype=torch.float16, device=GPU)
    t = fastest(lambda: [W @ X for _ in range(20)]) / 20
    flops = 2 * D * D * n                                    # one multiply and one add per weight per token
    nbytes = 2 * (D * D + 2 * D * n)                         # read W and X, write the result, 2 bytes each
    OUT['matmul'][n] = {'ms': t * 1e3, 'tflops': flops / t / 1e12, 'intensity': flops / nbytes}
    log(f'{n:>8} {t * 1e3:>8.3f} {flops / t / 1e12:>8.2f} {flops / nbytes:>15.1f}')
m = OUT['matmul']
log(f'from 1 to 64 tokens: time x{m[64]["ms"] / m[1]["ms"]:.1f} for 64x the work; '
    f'from 1 to 512: time x{m[512]["ms"] / m[1]["ms"]:.1f} for 512x the work')
del W

# ---------------------------------------------------------------- 4. batching (§2.3)
log('')
log('== 4. Request-level batching (pad to the longest) against iteration-level batching (Alpaca output lengths)')
alp = load_dataset('tatsu-lab/alpaca')['train']
rng = random.Random(0)
OUTLEN = [n for n in (len(tok(alp[i]['output']).input_ids) - 1 for i in rng.sample(range(len(alp)), 400)) if n > 0]
OUTLEN = OUTLEN[:300]
SLOTS, GAP = 8, 12                                          # 8 requests per batch; a new request every 12 steps
ARRIVE = [i * GAP for i in range(len(OUTLEN))]
log(f'{len(OUTLEN)} requests, one every {GAP} steps; output lengths: mean {statistics.mean(OUTLEN):.1f}, '
    f'shortest {min(OUTLEN)}, longest {max(OUTLEN)}; batch of {SLOTS}')


def request_level():
    """Method 1: take the waiting requests (up to 8) as one batch, run it until its longest answer ends, repeat.
    A request that ends early keeps its slot as padding; a request that arrives meanwhile waits."""
    t, i, waits, pad, busy = 0, 0, [], 0, 0
    batches = []
    while i < len(OUTLEN):
        t = max(t, ARRIVE[i])
        batch = [j for j in range(i, len(OUTLEN)) if ARRIVE[j] <= t][:SLOTS]
        i += len(batch)
        steps = max(OUTLEN[j] for j in batch)
        waits += [t - ARRIVE[j] for j in batch]
        pad += sum(steps - OUTLEN[j] for j in batch)
        busy += sum(OUTLEN[j] for j in batch)
        batches.append([OUTLEN[j] for j in batch])
        t += steps
    return {'steps': t, 'wait': statistics.mean(waits), 'pad': pad, 'useful': busy, 'batches': batches}


def iteration_level():
    """Method 1: after every step, finished requests leave and waiting requests take their slots at once."""
    t, i, running, waits, useful = 0, 0, [], [], 0
    while i < len(OUTLEN) or running:
        while i < len(OUTLEN) and ARRIVE[i] <= t and len(running) < SLOTS:
            waits.append(t - ARRIVE[i]); running.append(OUTLEN[i]); i += 1
        if not running:                                     # nothing to do until the next request arrives
            t = ARRIVE[i]; continue
        useful += len(running)
        running = [r - 1 for r in running if r > 1]         # one step: every running request writes one token
        t += 1
    return {'steps': t, 'wait': statistics.mean(waits), 'pad': 0, 'useful': useful}


R, I = request_level(), iteration_level()
pad_closed = sum(len(b) * max(b) - sum(b) for b in R['batches'])     # Method 2: per batch, size x longest - sum
log(f'{"":<16} {"steps to finish":>15} {"mean wait (steps)":>18} {"padding slot-steps":>19}')
log(f'{"request-level":<16} {R["steps"]:>15,} {R["wait"]:>18.1f} {R["pad"]:>19,}')
log(f'{"iteration-level":<16} {I["steps"]:>15,} {I["wait"]:>18.1f} {I["pad"]:>19,}')
log(f'useful slot-steps (one token each): {R["useful"]:,} in both; padding share of request-level batches '
    f'{100 * R["pad"] / (R["pad"] + R["useful"]):.1f}%; closed form for padding agrees: {pad_closed == R["pad"]}')
OUT['batching'] = {'request': {k: v for k, v in R.items() if k != 'batches'}, 'iteration': I,
                   'mean_out': statistics.mean(OUTLEN), 'lengths': OUTLEN, 'n': len(OUTLEN), 'gap': GAP, 'slots': SLOTS}

# ---------------------------------------------------------------- 5. large KV cache (§3)
log('')
log('== 5. KV cache size from the real configs (FP16, 2 bytes per number)')
log(f'{"model":<10} {"layers":>6} {"hidden":>6} {"KB per token":>12} {"GB per 2048 tokens":>18} {"weights GB":>10}')
OUT['sizes'] = {}
for name, params in (('opt-125m', 0.125), ('opt-1.3b', 1.3), ('opt-13b', 13), ('opt-66b', 66)):
    cfg = json.load(open(hf_hub_download(f'facebook/{name}', 'config.json')))
    per_tok = 2 * cfg['hidden_size'] * cfg['num_hidden_layers'] * 2
    per_req = per_tok * cfg['max_position_embeddings']
    OUT['sizes'][name] = {'layers': cfg['num_hidden_layers'], 'hidden': cfg['hidden_size'], 'kb': per_tok / 1024,
                          'gb_req': per_req / 1e9, 'weights_gb': params * 2}
    log(f'{name:<10} {cfg["num_hidden_layers"]:>6} {cfg["hidden_size"]:>6} {per_tok / 1024:>12,.0f} '
        f'{per_req / 1e9:>18.2f} {params * 2:>10.2f}')
fit = (80 - 26) / OUT['sizes']['opt-13b']['gb_req']
log(f'OPT-13B on an 80 GB GPU: {80 - 26} GB left after the weights = {fit:.0f} requests of 2048 tokens')
OUT['fit80'] = fit

# ---------------------------------------------------------------- 6. complex decoding (§3)
log('')
log('== 6. Parallel sampling and beam search with model.generate (OPT-125M)')
PROMPT = 'Write a Python function that returns the sum of a list of numbers.\n'
pids = tok(PROMPT, return_tensors='pt').input_ids
P = pids.shape[1]
with torch.no_grad():
    g = model.generate(pids, do_sample=True, top_p=0.95, max_new_tokens=40, num_return_sequences=4,
                       return_dict_in_generate=True, pad_token_id=tok.pad_token_id)
cache = g.past_key_values
k0 = cache.layers[0].keys                                   # (samples, heads, tokens, head size)
same = all(torch.equal(k0[0, :, :P], k0[s, :, :P]) for s in range(1, 4))
per_tok = 2 * model.config.hidden_size * model.config.num_hidden_layers * 4      # FP32 here: 4 bytes per number
total = sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
log(f'prompt of {P} tokens, 4 samples of up to 40 new tokens; cache shape of layer 0: {tuple(k0.shape)}')
log(f'the 4 copies of the prompt keys are identical: {same}')
log(f'cache held: {total:,} bytes; the prompt stored 4 times = {4 * P * per_tok:,} bytes, '
    f'of which 3 copies ({3 * P * per_tok:,} bytes, {100 * 3 * P / (4 * k0.shape[2]):.0f}% of the cache) are duplicates')
for s in range(4):
    log(f'  sample {s + 1}: {tok.decode(g.sequences[s, P:], skip_special_tokens=True)[:60]!r}')
OUT['sampling'] = {'prompt': P, 'cache_tokens': k0.shape[2], 'same': same, 'dup_share': 3 * P / (4 * k0.shape[2])}

with torch.no_grad():
    b = model.generate(pids, num_beams=4, num_return_sequences=4, max_new_tokens=30, do_sample=False,
                       early_stopping=True, pad_token_id=tok.pad_token_id)
beams = [row[P:].tolist() for row in b]
common = 0
while all(len(x) > common for x in beams) and len({x[common] for x in beams}) == 1:
    common += 1
prefixes = {tuple(x[:i + 1]) for x in beams for i in range(len(x))}    # a tree of prefixes: shared tokens once
separate = sum(len(x) for x in beams)
log(f'beam search, 4 beams, 30 new tokens: all 4 share their first {common} new tokens')
log(f'new tokens stored once per beam: {separate}; stored once per distinct prefix: {len(prefixes)} '
    f'({100 * (1 - len(prefixes) / separate):.0f}% less)')
for s, x in enumerate(beams):
    log(f'  beam {s + 1}: {tok.decode(x, skip_special_tokens=True)[:70]!r}')
OUT['beams'] = {'common': common, 'separate': separate, 'shared': len(prefixes)}

# ---------------------------------------------------------------- 7. unknown output lengths (§3)
log('')
log('== 7. The output length is unknown until the model stops')
Q = tok('Question: What is the capital of France?\nAnswer:', return_tensors='pt').input_ids
lens = []
for seed in range(20):
    torch.manual_seed(seed)
    with torch.no_grad():
        o = model.generate(Q, do_sample=True, top_p=0.95, max_new_tokens=200, pad_token_id=tok.pad_token_id)
    new = o[0, Q.shape[1]:].tolist()
    lens.append(new.index(tok.eos_token_id) + 1 if tok.eos_token_id in new else len(new))
log(f'20 sampled answers to one question (stop at </s> or 200 tokens): lengths {sorted(lens)}')
all_out = sorted(len(tok(alp[i]['output']).input_ids) - 1 for i in rng.sample(range(len(alp)), 2000))
q = lambda f: all_out[int(f * (len(all_out) - 1))]
log(f'2000 Alpaca answers: shortest {all_out[0]}, median {q(0.5)}, 90th percentile {q(0.9)}, longest {all_out[-1]} tokens')
OUT['lengths'] = {'samples': lens, 'alpaca': {'min': all_out[0], 'median': q(0.5), 'p90': q(0.9), 'max': all_out[-1]}}

# ---------------------------------------------------------------- 8. Figure 3
log('')
log('== 8. Figure 3: the slots of two pre-allocated requests at the current step')


def slots(max_len, prompt, generated, final):
    """Split one pre-allocated chunk into: slots that hold a token now, slots that a later token of this request
    will fill (reserved) and slots that no token will ever fill (internal fragmentation)."""
    now = prompt + generated
    return {'token states': now, 'reserved': final - now, 'never used': max_len - final}


A = dict(max_len=2048, prompt=7, generated=2, final=7 + 4)   # Four score ... our | fathers brought | forth <eos>
B = dict(max_len=512, prompt=3, generated=1, final=3 + 2)    # You only live | once | <eos>
OUT['fig3'] = {}
for name, r in (('A', A), ('B', B)):
    s = slots(**r)
    OUT['fig3'][name] = s
    log(f'request {name}: chunk of {r["max_len"]}: ' + ', '.join(f'{k} {v:,}' for k, v in s.items())
        + f'; useful share {100 * s["token states"] / r["max_len"]:.2f}%')
both = {k: OUT['fig3']['A'][k] + OUT['fig3']['B'][k] for k in OUT['fig3']['A']}
log(f'both chunks: {2048 + 512:,} slots; ' + ', '.join(f'{k} {v:,}' for k, v in both.items())
    + f'; even at the end only {11 + 5} slots ({100 * 16 / 2560:.2f}%) ever hold a token')

# ---------------------------------------------------------------- 9. compaction
log('')
log('== 9. Compaction means copying the KV cache: how fast does this laptop copy GPU memory?')
src = torch.empty(512 * 2 ** 20, dtype=torch.float16, device=GPU)   # 1 GiB
dst = torch.empty_like(src)
t = fastest(lambda: dst.copy_(src), repeats=7)
log(f'copy 1 GiB: {t * 1e3:.1f} ms = {2 * src.numel() / t / 1e9:.1f} GB copied per second')
move12 = 12e9 / (2 * src.numel() / t)
log(f'moving 12 GB of KV cache at that speed: {move12 * 1e3:.0f} ms, during which no request writes a token')
OUT['copy'] = {'ms_per_gib': t * 1e3, 'gb_per_s': 2 * src.numel() / t / 1e9, 'ms_12gb': move12 * 1e3}

save('part2', OUT)
log('')
log('saved results/part2.json')
