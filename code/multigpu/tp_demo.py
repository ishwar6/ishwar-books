"""Tensor parallelism for real, on one laptop.

Part A: split one MLP's two matrix multiplies Megatron-style (columns, then rows) and check the sum of the
        partial results equals the unsplit result; then show why splitting the first matrix by rows is wrong.
Part B: run Qwen2.5-0.5B with tensor parallelism across 2 separate processes that talk only through
        torch.distributed all_reduce (gloo backend, CPU). Each process holds half the attention heads,
        half the MLP and half of every KV-cache tensor. Compare logits and 24 greedy tokens with the
        unsplit Hugging Face model, and count the all-reduce calls and bytes.
Part C: time gloo all_reduce on this laptop for message sizes 4 B to 64 MiB, 2 and 4 processes, and fit
        time = alpha + n / beta. These are CPU processes on one machine: the numbers describe this laptop's
        shared-memory path, not NVLink. The point is the shape of the curve (a fixed cost plus a per-byte cost).
Run: python code/multigpu/tp_demo.py  -> results/tp_demo.json"""
import os, time, json, math, statistics
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from common import save, RES

MODEL = 'Qwen/Qwen2.5-0.5B'
PROMPT = 'Tensor parallelism splits every layer of a model across several GPUs, so'
NEW = 24


def part_a():
    torch.manual_seed(0)
    d, ffn, T = 896, 4864, 8                               # Qwen2.5-0.5B sizes, 8 tokens
    X = torch.randn(T, d, dtype=torch.float64)
    A = torch.randn(d, ffn, dtype=torch.float64) / d ** 0.5
    B = torch.randn(ffn, d, dtype=torch.float64) / ffn ** 0.5
    full = F.gelu(X @ A) @ B
    res = {}
    for p in [2, 4, 8]:
        A_parts = A.chunk(p, dim=1)                        # column split: each GPU gets ffn/p output columns
        B_parts = B.chunk(p, dim=0)                        # row split: each GPU gets the matching ffn/p rows
        partial = [F.gelu(X @ Ai) @ Bi for Ai, Bi in zip(A_parts, B_parts)]   # no communication needed here
        Y = sum(partial)                                   # the all-reduce: add the p partial outputs
        res[p] = float((Y - full).abs().max())
        print(f'p={p}: each part A_i {tuple(A_parts[0].shape)}, B_i {tuple(B_parts[0].shape)}; '
              f'max |sum of parts - unsplit| = {res[p]:.1e}')
    # The wrong way: split A by rows (and X by columns). GeLU of a partial sum is not the sum of GeLUs.
    Xs, As = X.chunk(2, dim=1), A.chunk(2, dim=0)
    wrong = sum(F.gelu(Xi @ Ai) for Xi, Ai in zip(Xs, As)) @ B
    err = float((wrong - full).abs().max())
    print(f'row-split first matrix, GeLU applied before adding: max error {err:.3f} '
          f'(output values are about {float(full.abs().mean()):.3f} on average) -> needs a sync before GeLU')
    return dict(max_err=res, wrong_split_err=err, mean_abs_output=float(full.abs().mean()))


# ---------------------------------------------------------------- Part B: a real 2-process TP forward
def rope(x, pos, theta, hd):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    f = torch.outer(pos.float(), inv)
    emb = torch.cat([f, f], dim=-1)
    cos, sin = emb.cos(), emb.sin()
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


class Shard:
    """This rank's slice of every layer: its query heads, its KV heads, its share of the MLP."""
    def __init__(self, sd, cfg, rank, world):
        self.cfg, self.rank, self.world = cfg, rank, world
        self.theta = cfg.rope_parameters['rope_theta']
        H, Hkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.hidden_size // cfg.num_attention_heads
        self.hq, self.hkv, self.hd = H // world, Hkv // world, hd
        ffn = cfg.intermediate_size // world
        qs, ks = slice(rank * self.hq * hd, (rank + 1) * self.hq * hd), slice(rank * self.hkv * hd, (rank + 1) * self.hkv * hd)
        fs = slice(rank * ffn, (rank + 1) * ffn)
        self.layers = []
        for i in range(cfg.num_hidden_layers):
            g = lambda n: sd[f'model.layers.{i}.{n}']
            self.layers.append(dict(
                ln1=g('input_layernorm.weight'), ln2=g('post_attention_layernorm.weight'),
                wq=g('self_attn.q_proj.weight')[qs], bq=g('self_attn.q_proj.bias')[qs],      # column split
                wk=g('self_attn.k_proj.weight')[ks], bk=g('self_attn.k_proj.bias')[ks],
                wv=g('self_attn.v_proj.weight')[ks], bv=g('self_attn.v_proj.bias')[ks],
                wo=g('self_attn.o_proj.weight')[:, qs],                                        # row split
                wg=g('mlp.gate_proj.weight')[fs], wu=g('mlp.up_proj.weight')[fs],              # column split
                wd=g('mlp.down_proj.weight')[:, fs]))                                          # row split
        self.emb, self.norm = sd['model.embed_tokens.weight'], sd['model.norm.weight']
        self.cache = [dict(k=None, v=None) for _ in self.layers]
        self.n_allreduce, self.bytes_allreduce, self.t_allreduce = 0, 0, 0.0

    def all_reduce(self, x):
        if self.world == 1:                              # one process: nothing to add up, no call
            return x
        t = time.perf_counter()
        dist.all_reduce(x)                               # every rank ends with the sum of all partial outputs
        self.t_allreduce += time.perf_counter() - t
        self.n_allreduce += 1
        self.bytes_allreduce += x.numel() * x.element_size()
        return x

    def forward(self, ids, start):
        c = self.cfg
        h = self.emb[ids]                                # (T, d): embeddings are replicated on every rank
        pos = torch.arange(start, start + ids.shape[0])
        for L, kv in zip(self.layers, self.cache):
            x = rms(h, L['ln1'], c.rms_norm_eps)
            q = (x @ L['wq'].T + L['bq']).view(-1, self.hq, self.hd).transpose(0, 1)
            k = (x @ L['wk'].T + L['bk']).view(-1, self.hkv, self.hd).transpose(0, 1)
            v = (x @ L['wv'].T + L['bv']).view(-1, self.hkv, self.hd).transpose(0, 1)
            q, k = rope(q, pos, self.theta, self.hd), rope(k, pos, self.theta, self.hd)
            kv['k'] = k if kv['k'] is None else torch.cat([kv['k'], k], 1)      # this rank caches only its KV heads
            kv['v'] = v if kv['v'] is None else torch.cat([kv['v'], v], 1)
            K = kv['k'].repeat_interleave(self.hq // self.hkv, 0)
            V = kv['v'].repeat_interleave(self.hq // self.hkv, 0)
            a = F.scaled_dot_product_attention(q, K, V, is_causal=ids.shape[0] > 1)
            a = a.transpose(0, 1).reshape(ids.shape[0], -1)
            h = h + self.all_reduce(a @ L['wo'].T)       # all-reduce 1 of 2 in this layer
            x = rms(h, L['ln2'], c.rms_norm_eps)
            m = F.silu(x @ L['wg'].T) * (x @ L['wu'].T)
            h = h + self.all_reduce(m @ L['wd'].T)       # all-reduce 2 of 2 in this layer
        return rms(h, self.norm, c.rms_norm_eps) @ self.emb.T   # tied output head, replicated

    def kv_bytes(self):
        return sum(t.numel() * t.element_size() for kv in self.cache for t in kv.values())


def worker(rank, world, port, q):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    torch.set_num_threads(4)
    from transformers import AutoConfig, AutoTokenizer
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL)
    cfg = AutoConfig.from_pretrained(path)
    sd = {k: v.float() for k, v in load_file(os.path.join(path, 'model.safetensors')).items()}
    s = Shard(sd, cfg, rank, world)
    del sd
    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(PROMPT, return_tensors='pt').input_ids[0]
    dist.barrier()
    with torch.no_grad():
        logits = s.forward(ids, 0)
        prefill_calls, prefill_bytes = s.n_allreduce, s.bytes_allreduce
        out, nxt, t_dec = [], int(logits[-1].argmax()), []
        for i in range(NEW):
            out.append(nxt)
            n0, b0 = s.n_allreduce, s.bytes_allreduce
            t = time.perf_counter()
            l = s.forward(torch.tensor([nxt]), ids.shape[0] + i)
            t_dec.append(time.perf_counter() - t)
            nxt = int(l[-1].argmax())
        per_tok = dict(calls=s.n_allreduce - n0, bytes=s.bytes_allreduce - b0)
    params_here = sum(t.numel() for L in s.layers for t in L.values())
    if rank == 0:
        torch.save(dict(world=world, logits=logits[-1].tolist(), tokens=out, prefill_tokens=int(ids.shape[0]),
                   prefill_allreduce_calls=prefill_calls, prefill_allreduce_bytes=prefill_bytes,
                   decode_allreduce_per_token=per_tok, layer_params_per_rank=params_here,
                   kv_bytes_rank0=s.kv_bytes(), decode_ms_median=statistics.median(t_dec) * 1e3,
                   allreduce_s_total=s.t_allreduce, text=tok.decode(out)), q)
    dist.destroy_process_group()


def part_b():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    ref = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    ids = tok(PROMPT, return_tensors='pt').input_ids
    with torch.no_grad():
        ref_logits = ref(ids).logits[0, -1]
        gen = ref.generate(ids, max_new_tokens=NEW, do_sample=False)[0, ids.shape[1]:].tolist()
    layer_params = sum(p.numel() for n, p in ref.named_parameters() if '.layers.' in n)
    res = {}
    for world in [1, 2]:
        f = str(RES / f'.tp{world}.pt')               # rank 0 writes its result here (a pipe can deadlock on big objects)
        mp.spawn(worker, args=(world, 29500 + world, f), nprocs=world, join=True)
        r = torch.load(f); os.remove(f)
        diff = float((torch.tensor(r.pop('logits')) - ref_logits).abs().max())
        r.update(max_logit_diff_vs_hf=diff, same_tokens_as_hf=r['tokens'] == gen,
                 layer_params_fraction=r['layer_params_per_rank'] / layer_params)
        res[world] = r
        pt = r['decode_allreduce_per_token']
        print(f"TP={world}: rank 0 holds {r['layer_params_fraction']:.1%} of the layer weights and "
              f"{r['kv_bytes_rank0']/1024:.0f} KiB of KV cache; max |logit diff vs HF| = {diff:.1e}; "
              f"same {NEW} greedy tokens as HF: {r['same_tokens_as_hf']}")
        print(f"      all-reduces: prefill of {r['prefill_tokens']} tokens = {r['prefill_allreduce_calls']} calls, "
              f"{r['prefill_allreduce_bytes']/1024:.0f} KiB; each decode token = {pt['calls']} calls, "
              f"{pt['bytes']/1024:.1f} KiB; decode step median {r['decode_ms_median']:.1f} ms")
    print('generated:', repr(res[2]['text']))
    return {str(k): v for k, v in res.items()}


# ---------------------------------------------------------------- Part C: all-reduce cost on this laptop
def bench(rank, world, port, q):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
    dist.init_process_group('gloo', rank=rank, world_size=world)
    torch.set_num_threads(1)
    rows = []
    for e in range(0, 25, 2):                            # 1 to 16M float32 elements = 4 B to 64 MiB
        n = 2 ** e
        x = torch.ones(n)
        reps = 100 if n < 2 ** 14 else (20 if n < 2 ** 20 else 4)
        for _ in range(3):
            dist.all_reduce(x)
        loops = []
        for _ in range(7):                               # time a back-to-back loop, keep the fastest loop
            dist.barrier()
            t = time.perf_counter()
            for _ in range(reps):
                dist.all_reduce(x)
            loops.append((time.perf_counter() - t) / reps)
        rows.append(dict(bytes=4 * n, us=min(loops) * 1e6))   # fastest loop: least disturbed by other jobs
    if rank == 0:
        torch.save(rows, q)
    dist.destroy_process_group()


def part_c():
    out = {}
    for world in [2, 4]:
        f = str(RES / f'.ar{world}.pt')
        mp.spawn(bench, args=(world, 29600 + world, f), nprocs=world, join=True)
        rows = torch.load(f); os.remove(f)
        alpha = statistics.median(r['us'] for r in rows if r['bytes'] <= 1024)   # small messages: fixed cost only
        a, b = rows[-2], rows[-1]
        beta = (b['bytes'] - a['bytes']) / ((b['us'] - a['us']) * 1e-6)         # slope between the two largest sizes
        out[world] = dict(rows=rows, alpha_us=alpha, beta_GBps=beta / 1e9)
        print(f'gloo all_reduce, {world} processes on this laptop (CPU, shared memory):')
        for r in rows:
            model = alpha + r['bytes'] / beta * 1e6
            print(f"   {r['bytes']:>10,} B  {r['us']:10.1f} us   (fit alpha + n/beta: {model:10.1f} us)")
        print(f'   fitted alpha = {alpha:.1f} us, beta = {beta/1e9:.2f} GB/s')
    return {str(k): v for k, v in out.items()}


if __name__ == '__main__':
    print('PART A: split matrix multiplies (float64, CPU)')
    a = part_a()
    print('\nPART B: Qwen2.5-0.5B, tensor parallel across real processes (float32, CPU, gloo)')
    b = part_b()
    print('\nPART C: what one all-reduce costs on this laptop')
    c = part_c()
    save('tp_demo', dict(split_matmul=a, qwen_tp=b, allreduce_bench=c, torch=torch.__version__))
