"""Chapter 8, step 1: choose the prompts for PPO.

UltraFeedback prompts (single turn, at most 128 tokens), none of which the Chapter 7 reward model was trained or tested on.
Qwen2.5-0.5B-Instruct writes long answers: at 256 new tokens only about a third of its replies end. To keep each PPO step
short we cap replies at 128 tokens and keep the prompts for which one sample from the starting model (temperature 0.7)
finished within that cap. Training prompts come from the train split, evaluation prompts from the test split."""
import json, random, time
import torch
from common import Log, save
from ch7_common import parquet, load_pairs, tok
from ch8_common import load_policy, generate, PROMPTS

MAX_NEW, TEMP, BS = 128, 0.7, 48
log = Log('ch8_prompts')
rm_seen = {p['prompt'] for p in load_pairs('train_prefs', 4000, seed=0)} | {p['prompt'] for p in load_pairs('test_prefs', 600, seed=123)}


def candidates(split, n, seed):
    d = parquet(split)
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    out, seen = [], set()
    for i in idx:
        p = d.iloc[i].prompt
        if p in rm_seen or p in seen or not p.strip() or len(tok(p)['input_ids']) > 128:
            continue
        seen.add(p); out.append(p)
        if len(out) == n:
            break
    return out


model = load_policy()                              # fresh LoRA adapter: B = 0, identical to the starting model
torch.manual_seed(0)
res = {}
for split, n in [("test_prefs", 200), ("train_prefs", 1200)]:
    cand = candidates(split, n, seed=8)
    kept, lens, t0 = [], [], time.time()
    for k in range(0, len(cand), BS):
        for s in generate(model, cand[k:k + BS], MAX_NEW, TEMP):
            lens.append(len(s['r_ids']) if s['finished'] else None)
            if s['finished']:
                kept.append(s['prompt'])
        if (k // BS) % 10 == 0:
            log(f'{split}: {k + BS} candidates, kept {len(kept)}, {time.time() - t0:.0f}s')
    fin = [l for l in lens if l is not None]
    log(f'{split}: {len(cand)} candidates, {len(kept)} kept ({len(kept) / len(cand):.2f}); '
        f'finished replies: mean {sum(fin) / len(fin):.1f} tokens, median {sorted(fin)[len(fin) // 2]}')
    res[split] = dict(candidates=len(cand), kept=len(kept), mean_len=sum(fin) / len(fin))
    res[split + '_prompts'] = kept
json.dump(dict(train=res['train_prefs_prompts'], test=res['test_prefs_prompts']), open(PROMPTS, 'w'))
save('ch8_prompts', {k: v for k, v in res.items() if not k.endswith('_prompts')})
