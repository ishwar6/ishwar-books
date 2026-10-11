"""Chapter 7, overoptimisation experiment, step 1: sample N answers per prompt from the policy we would later train
(Qwen2.5-0.5B-Instruct). Held-out UltraFeedback prompts, temperature 1.0, up to 256 new tokens.
The samples go to the cache folder (they are large); step 2 (ch7_bon.py) scores them."""
import json, os, random, time
import torch
from transformers import AutoModelForCausalLM
from common import Log
from ch7_common import parquet, tok, BACKBONE, DEV, CACHE, load_pairs

N_PROMPTS, N, MAX_NEW = 100, 32, 256
log = Log('ch7_bon_gen')
test_prompts = {p['prompt'] for p in load_pairs('test_prefs', 600, seed=123)}   # keep the RM test pairs out
d = parquet('test_prefs')
idx = list(range(len(d)))
random.Random(7).shuffle(idx)
prompts = []
for i in idx:
    p = d.iloc[i].prompt
    if p in test_prompts or p in prompts:
        continue
    if len(tok(p)['input_ids']) <= 150:
        prompts.append(p)
    if len(prompts) == N_PROMPTS:
        break
log(f'{len(prompts)} prompts, {N} samples each, temperature 1.0, max {MAX_NEW} new tokens')

model = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.bfloat16).to(DEV).eval()
tok.padding_side = 'left'
stop = [tok.convert_tokens_to_ids(t) for t in ['<|endoftext|>', '<|im_end|>']]
torch.manual_seed(0)
out, t0 = [], time.time()
for k, p in enumerate(prompts):
    text = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
    enc = tok([text] * N, return_tensors='pt', padding=True).to(DEV)
    with torch.no_grad():
        g = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, temperature=1.0, top_p=1.0, top_k=None,
                           eos_token_id=stop, pad_token_id=tok.pad_token_id)
    resp, done = [], []
    for row in g[:, enc['input_ids'].shape[1]:].tolist():
        cut = next((j for j, t in enumerate(row) if t in stop), None)
        done.append(cut is not None)
        resp.append(tok.decode(row[:cut if cut is not None else len(row)]))
    out.append(dict(prompt=p, responses=resp, finished=done))
    if (k + 1) % 10 == 0:
        log(f'  {k + 1} prompts, {time.time() - t0:.0f}s, finished share so far {sum(sum(o["finished"]) for o in out) / (len(out) * N):.2f}')
        json.dump(out, open(os.path.join(CACHE, 'bon_samples.json'), 'w'))
json.dump(out, open(os.path.join(CACHE, 'bon_samples.json'), 'w'))
log(f'done in {time.time() - t0:.0f}s')
