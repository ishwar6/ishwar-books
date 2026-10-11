"""Chapter 8: evaluate PPO checkpoints on held-out prompts with the proxy (our Chapter 7 reward model) AND a gold judge
(Skywork-Reward-V2-Qwen3-0.6B), as Chapter 7 did for best-of-n.

For every checkpoint: 2 replies to each held-out prompt (temperature 0.7, at most 128 tokens, the training settings),
then proxy score, gold score, length, share finished, KL to the starting model, and distinct-2.
Usage: python ch8_eval.py beta005 nokl ...   (results/ch8_eval.json keeps one entry per run)"""
import json, os, sys, time
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from common import Log, save, RESULTS
from ch7_common import load_rm, score_ids, load_gold, score_gold, tok, BACKBONE, DEV
from ch8_common import load_policy, generate, batch_tensors, token_logprobs, rm_texts, load_prompts, CKPT

runs = sys.argv[1:]
log = Log('ch8_eval_' + '_'.join(runs))
TEMP, MAX_NEW, N = 0.7, 128, 2
cal = json.load(open(os.path.join(RESULTS, 'ch8_calib.json')))
test = load_prompts()['test']
log(f'{len(test)} held-out prompts x {N} replies per checkpoint')
rm = load_rm()
gt, gm = load_gold()
OUT = os.path.join(RESULTS, 'ch8_eval.json')
allres = json.load(open(OUT)) if os.path.exists(OUT) else {}


def distinct2(texts):
    grams = [tuple(w[i:i + 2]) for t in texts for w in [t.split()] for i in range(len(w) - 1)]
    return len(set(grams)) / max(1, len(grams))


def evaluate(model, use_adapter):
    torch.manual_seed(1234)                       # the same sampling noise for every checkpoint
    S = []
    for k in range(0, len(test), 32):
        if use_adapter:
            S += generate(model, test[k:k + 32], MAX_NEW, TEMP, n=N)
        else:
            with model.disable_adapter():
                S += generate(model, test[k:k + 32], MAX_NEW, TEMP, n=N)
    kl = []
    with torch.no_grad():
        for k in range(0, len(S), 8):
            ids, att, act = batch_tensors(S[k:k + 8])
            if use_adapter:
                lp = token_logprobs(model, ids, att, TEMP)
            else:
                with model.disable_adapter():
                    lp = token_logprobs(model, ids, att, TEMP)
            with model.disable_adapter():
                ref = token_logprobs(model, ids, att, TEMP)
            kl += ((lp - ref) * act).sum(1).tolist()
    texts = [tok.decode(s['r_ids']) for s in S]
    proxy = (np.array(score_ids(rm, rm_texts(S))) - cal['mean']) / cal['sd']
    gold = np.array(score_gold(gt, gm, [s['prompt'] for s in S], texts))
    fin = np.array([s['finished'] for s in S])
    return dict(proxy=proxy.tolist(), gold=gold.tolist(), finished=fin.tolist(), kl=kl,
                length=[len(s['r_ids']) for s in S], words=[len(t.split()) for t in texts],
                distinct2=distinct2(texts), texts=texts, prompts=[s['prompt'] for s in S])


# the starting model, once (shared by all runs)
if 'start' not in allres:
    m = load_policy().eval()                      # a fresh adapter with B = 0; evaluated with the adapter switched off
    t0 = time.time()
    allres['start'] = evaluate(m, use_adapter=False)
    json.dump(allres, open(OUT, 'w'))
    log(f'start evaluated in {time.time() - t0:.0f}s')
    del m
st = allres['start']
g_mu, g_sd = float(np.mean(st['gold'])), float(np.std(st['gold']))
allres['gold_norm'] = dict(mean=g_mu, sd=g_sd)


def summary(name, r):
    p, g, f = np.array(r['proxy']), (np.array(r['gold']) - g_mu) / g_sd, np.array(r['finished'])
    log(f'  {name:16s} proxy {p.mean():+.3f}  gold {g.mean():+.3f} (raw {np.mean(r["gold"]):+.3f})  KL {np.mean(r["kl"]):6.2f}  '
        f'tokens {np.mean(r["length"]):5.1f}  words {np.mean(r["words"]):5.1f}  finished {f.mean():.2f}  distinct-2 {r["distinct2"]:.3f}')


log('gold scores are standardised with the starting model\'s mean and sd on these replies '
    f'(raw mean {g_mu:+.3f}, sd {g_sd:.3f})')
summary('start', st)
for run in runs:
    steps = sorted(int(d[4:]) for d in os.listdir(os.path.join(CKPT, run)) if d.startswith('step'))
    allres.setdefault(run, {})
    for s in steps:
        if str(s) in allres[run]:
            summary(f'{run} step {s}', allres[run][str(s)]); continue
        base_ = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32)
        m = PeftModel.from_pretrained(base_, os.path.join(CKPT, run, f'step{s}')).to(DEV).eval()
        t0 = time.time()
        allres[run][str(s)] = evaluate(m, use_adapter=True)
        summary(f'{run} step {s}', allres[run][str(s)])
        json.dump(allres, open(OUT, 'w'))
        del m, base_
log('done')
