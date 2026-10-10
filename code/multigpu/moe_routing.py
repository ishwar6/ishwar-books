"""Measure real expert routing in a real Mixture-of-Experts model, then ask what it means for expert parallelism.

Model: OLMoE-1B-7B-0924 (64 experts per layer, 8 chosen per token, 16 layers), BF16 on the Apple GPU (mps).
Text: WikiText-2 (test split) and Python code from codeparrot-clean-valid. We record, for every token and layer,
which 8 experts the router picked. Nothing here times anything: the routing decisions are the measurement.
Then, in plain Python (simulation on the recorded decisions):
  - load imbalance: the busiest expert / GPU against the average, for batch sizes and EP sizes;
  - the same for random routing, to separate bad luck from real skew;
  - redundant experts (DeepSeek-V3, Section 3.4): copy the hottest experts chosen on one half of the
    text and measure the busiest GPU on the other half;
Run: python code/multigpu/moe_routing.py  -> results/moe_routing.json and results/moe_picks.npz
     python code/multigpu/moe_routing.py --reuse   (re-analyse the saved routing decisions, no model needed)"""
import random, statistics, sys, time
import numpy as np
import torch
from common import save, RES

MODEL = 'allenai/OLMoE-1B-7B-0924'
SEQ, NSEQ = 512, 24                                   # 24 sequences of 512 tokens per text source
E, K = 64, 8


def load_texts(tok):
    from datasets import load_dataset
    wiki = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    wiki_ids = tok('\n'.join(t for t in wiki['text'] if t.strip()), return_tensors='np').input_ids[0]
    code = load_dataset('codeparrot/codeparrot-clean-valid', split='train[:400]')
    code_ids = tok('\n\n'.join(code['content'][:400]), return_tensors='np').input_ids[0]
    cut = lambda ids: [ids[i * SEQ:(i + 1) * SEQ] for i in range(NSEQ)]
    return {'wikitext': cut(wiki_ids), 'python code': cut(code_ids)}


def record(model, seqs, dev):
    picks = []                                          # list over layers of (tokens, 8) expert ids
    hooks, buf = [], {}
    for i, layer in enumerate(model.model.layers):
        hooks.append(layer.mlp.gate.register_forward_hook(lambda m, inp, out, i=i: buf.setdefault(i, []).append(out[2].cpu())))
    with torch.no_grad():
        for s in seqs:
            model(torch.tensor(s, device=dev)[None])
    for h in hooks:
        h.remove()
    return np.stack([torch.cat(buf[i]).numpy() for i in range(len(model.model.layers))])   # (layers, tokens, 8)


def gpu_of(n_gpus, placement=None):
    return placement if placement is not None else np.arange(E) // (E // n_gpus)


def imbalance(picks, batch, n_gpus, rng, placement=None, replicas=None):
    """Busiest GPU / average GPU, averaged over random batches of `batch` tokens and over layers.
    placement: expert -> GPU. replicas: expert -> list of GPUs sharing its tokens (split evenly)."""
    L, T, _ = picks.shape
    ratios = []
    for _ in range(40):
        idx = rng.choice(T, batch, replace=False)
        for l in range(L):
            counts = np.bincount(picks[l, idx].ravel(), minlength=E).astype(float)
            load = np.zeros(n_gpus)
            if replicas is None:
                np.add.at(load, gpu_of(n_gpus, placement), counts)
            else:
                for e, gpus in replicas.items():
                    for g in gpus:
                        load[g] += counts[e] / len(gpus)
            ratios.append(load.max() / load.mean())
    return float(np.mean(ratios))


def random_routing(L, T, rng):
    return np.stack([np.stack([rng.choice(E, K, replace=False) for _ in range(T)]) for _ in range(L)])


def balanced_placement(hist, n_gpus):
    """Place experts by past load: heaviest first, each onto the least-loaded GPU that still has a free slot."""
    freq = np.bincount(hist.ravel(), minlength=E).astype(float)
    per, load, slots = E // n_gpus, np.zeros(n_gpus), np.full(n_gpus, E // n_gpus)
    place = np.zeros(E, dtype=int)
    for e in np.argsort(-freq):
        g = min((g for g in range(n_gpus) if slots[g] > 0), key=lambda g: load[g])
        place[e] = g; load[g] += freq[e]; slots[g] -= 1
    return place


def add_redundant(hist, n_gpus, n_extra, place):
    """DeepSeek-style redundant experts, greedy: up to n_extra times, take the busiest GPU (by past load), copy its
    busiest expert onto the least-loaded GPU that has a free slot and lacks that expert, and split that expert's
    tokens evenly between its copies. A copy is kept only if its new GPU stays below the old peak; otherwise
    try the next expert or the next-busiest GPU. Each GPU has n_extra / n_gpus free slots."""
    freq = np.bincount(hist.ravel(), minlength=E).astype(float)
    replicas = {e: [int(place[e])] for e in range(E)}
    load = np.zeros(n_gpus)
    np.add.at(load, place, freq)
    slots = np.full(n_gpus, n_extra // n_gpus)
    for _ in range(n_extra):
        done = False
        for h in np.argsort(-load):                    # busiest GPU first
            here = sorted((e for e in range(E) if h in replicas[e]), key=lambda e: -freq[e] / len(replicas[e]))
            for e in here:                             # its busiest expert first
                cand = [g for g in range(n_gpus) if slots[g] > 0 and g not in replicas[e]]
                if not cand:
                    continue
                g = min(cand, key=lambda g: load[g])
                old = len(replicas[e])
                new = load.copy()
                for k in replicas[e]:
                    new[k] -= freq[e] / old - freq[e] / (old + 1)
                new[g] += freq[e] / (old + 1)
                if new[g] < load[h]:                   # the copy must not create a new busiest GPU
                    load = new; replicas[e].append(g); slots[g] -= 1; done = True
                    break
            if done:
                break
        if not done:
            break
    return replicas


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL)
    rng = np.random.default_rng(0)
    out = {}
    cache = RES / 'moe_picks.npz'
    if '--reuse' in sys.argv and cache.exists():      # analyse the saved routing decisions without the model
        z = np.load(cache)
        recorded = {k.replace('_', ' '): z[k].astype(np.int64) for k in z.files}
        print(f'Routing decisions loaded from {cache.name} (recorded earlier with OLMoE-1B-7B on {dev})')
    else:
        t0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
        print(f'OLMoE-1B-7B loaded on {dev} in {time.time()-t0:.0f} s: {sum(p.numel() for p in model.parameters())/1e9:.2f}B parameters, '
              f'{len(model.model.layers)} layers x {E} experts, top-{K}')
        texts = load_texts(tok)
        recorded = {name: record(model, seqs, dev) for name, seqs in texts.items()}
        np.savez_compressed(cache, **{k.replace(' ', '_'): v.astype(np.int8) for k, v in recorded.items()})
    for name, picks in recorded.items():
        L, T, _ = picks.shape
        freq = np.stack([np.bincount(picks[l].ravel(), minlength=E) for l in range(L)]) / (T * K)   # share of visits
        hot = freq.max(1) * E                                      # hottest expert vs a fair 1/64 share
        cold = freq.min(1) * E
        print(f'\n{name}: {T:,} tokens x {L} layers recorded')
        print(f'  hottest expert per layer gets {np.median(hot):.1f}x its fair share (median over layers, range '
              f'{hot.min():.1f}x to {hot.max():.1f}x); coldest gets {np.median(cold):.2f}x')
        res = dict(tokens=int(T), layers=int(L), share=freq.round(5).tolist(), hottest_x=hot.tolist(), coldest_x=cold.tolist())
        res['imbalance'] = {}
        rnd = random_routing(L, T, rng)
        for G in [8, 16, 64]:
            for B in [64, 256, 4096]:
                real = imbalance(picks, B, G, rng)
                ideal = imbalance(rnd, B, G, rng)
                res['imbalance'][f'EP{G}_B{B}'] = dict(real=real, random=ideal)
        print('  busiest GPU / average GPU (1.00 = perfect).  columns: tokens in the batch')
        print('             ' + ''.join(f'{f"B={B}":>18}' for B in [64, 256, 4096]))
        for G in [8, 16, 64]:
            cells = ''.join(f"{res['imbalance'][f'EP{G}_B{B}']['real']:8.2f} (rand {res['imbalance'][f'EP{G}_B{B}']['random']:.2f})" for B in [64, 256, 4096])
            print(f'  EP={G:<3} real {cells}')
        # Redundant experts: learn hot experts on the first half of the tokens, test on the second half.
        half = T // 2
        hist, test = picks[:, :half], picks[:, half:]
        red = {}
        for G, extra in [(8, 8), (8, 16), (16, 16), (64, 64)]:
            b0, b1, b2 = [], [], []
            for l in range(L):
                place = balanced_placement(hist[l], G)
                reps = add_redundant(hist[l], G, extra, place)
                b0.append(imbalance(test[l:l + 1], 4096, G, rng))
                b1.append(imbalance(test[l:l + 1], 4096, G, rng, placement=place))
                b2.append(imbalance(test[l:l + 1], 4096, G, rng, replicas=reps))
            red[f'EP{G}+{extra}'] = dict(default=float(np.mean(b0)), balanced=float(np.mean(b1)), redundant=float(np.mean(b2)))
            print(f'  EP={G:<2} placement learned on the first half, tested on the second (B=4096): in order {np.mean(b0):.2f}, '
                  f'balanced by load {np.mean(b1):.2f}, plus {extra} redundant copies {np.mean(b2):.2f}')
        res['redundant'] = red
        out[name] = res
    # Do the two sources agree on which experts are hot?
    a, b = np.array(out['wikitext']['share']), np.array(out['python code']['share'])
    top = lambda s: set(np.argsort(-s)[:8])
    overlap = [len(top(a[l]) & top(b[l])) for l in range(a.shape[0])]
    print(f'\nTop-8 hottest experts shared by wikitext and code: median {statistics.median(overlap)} of 8 per layer '
          f'(range {min(overlap)} to {max(overlap)})')
    out['hot_overlap_wiki_code'] = overlap
    save('moe_routing', out)


if __name__ == '__main__':
    main()
