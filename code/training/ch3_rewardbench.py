"""Chapter 3, experiment 8: evaluating reward models the RewardBench way.
A reward model passes a test item if it gives the chosen answer a higher score than the rejected one.
We take 100 random pairs from each of RewardBench's four sections (pooled over the section's subsets; the official
score weights subsets differently, so our numbers are a simplified version) and score them with three reward models:
  OA DeBERTa (2023)    OpenAssistant/reward-model-deberta-v3-large-v2, a 435M classifier (inputs cut at 512 tokens)
  Skywork V2 (2025)    Skywork/Skywork-Reward-V2-Qwen3-0.6B, a 0.6B classifier (inputs cut at 2048 tokens)
  implicit (DPO-style) log P_instruct(answer) - log P_base(answer) with Qwen2.5-0.5B, the reward a DPO-trained
                       policy implies (Rafailov et al. 2023); RewardBench scores DPO models exactly this way
Also: does each reward model simply prefer the longer answer?"""
import torch
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import Log, save
from ch3_common import BASE, INSTRUCT, DEV, data, load, sample_rows
import pandas as pd

log = Log('ch3_rewardbench')
SECTIONS = {
    'Chat': ['alpacaeval-easy', 'alpacaeval-length', 'alpacaeval-hard', 'mt-bench-easy', 'mt-bench-med'],
    'Chat Hard': ['mt-bench-hard', 'llmbar-natural', 'llmbar-adver-neighbor', 'llmbar-adver-GPTInst',
                  'llmbar-adver-GPTOut', 'llmbar-adver-manual'],
    'Safety': ['refusals-dangerous', 'refusals-offensive', 'xstest-should-refuse', 'xstest-should-respond', 'donotanswer'],
    'Reasoning': ['math-prm', 'hep-cpp', 'hep-go', 'hep-java', 'hep-js', 'hep-python', 'hep-rust'],
}
rb = data('rewardbench')
pairs = []
for sec, subs in SECTIONS.items():
    d = sample_rows(rb[rb.subset.isin(subs)], 100, seed=0)
    pairs += [dict(section=sec, subset=r.subset, prompt=r.prompt, chosen=r.chosen, rejected=r.rejected) for r in d.itertuples()]
log(f'{len(pairs)} pairs: ' + ', '.join(f'{s} {sum(p["section"] == s for p in pairs)}' for s in SECTIONS))
log(f'chosen answer is the longer one in {sum(len(p["chosen"]) > len(p["rejected"]) for p in pairs)} of {len(pairs)} pairs')


@torch.no_grad()
def score_deberta(items):
    name = 'OpenAssistant/reward-model-deberta-v3-large-v2'
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).to(DEV).eval()
    out = []
    for q, a in items:
        x = tok(q, a, return_tensors='pt', truncation=True, max_length=512).to(DEV)
        out.append(m(**x).logits[0, 0].item())
    return out


@torch.no_grad()
def score_skywork(items):
    name = 'Skywork/Skywork-Reward-V2-Qwen3-0.6B'
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, dtype=torch.float32, num_labels=1).to(DEV).eval()
    out = []
    for q, a in items:
        conv = [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
        ids = tok.apply_chat_template(conv, tokenize=True, return_tensors='pt', return_dict=True)['input_ids'][:, -2048:].to(DEV)
        out.append(m(input_ids=ids).logits[0, 0].item())
    return out


@torch.no_grad()
def answer_logprob(tok, model, q, a):
    p = tok.apply_chat_template([{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)
    pi = tok(p)['input_ids'][-1024:]
    ai = tok(a + '<|im_end|>')['input_ids'][:1024]
    x = torch.tensor([pi + ai], device=DEV)
    lp = Fn.log_softmax(model(x).logits[0, len(pi) - 1:-1].float(), -1)
    return lp.gather(1, x[0, len(pi):, None]).sum().item()


def score_implicit(items):
    lps = {}
    for name in [INSTRUCT, BASE]:
        tok, model = load(name)
        lps[name] = [answer_logprob(tok, model, q, a) for q, a in items]
        del model
    return [i - b for i, b in zip(lps[INSTRUCT], lps[BASE])]


items = [(p['prompt'], p['chosen']) for p in pairs] + [(p['prompt'], p['rejected']) for p in pairs]
R = {'pairs': [{k: p[k] for k in ['section', 'subset']} | dict(len_chosen=len(p['chosen']), len_rejected=len(p['rejected']))
               for p in pairs], 'rms': {}}
n = len(pairs)
for label, fn in [('OA DeBERTa (2023)', score_deberta), ('Skywork V2 (2025)', score_skywork), ('implicit (DPO-style)', score_implicit)]:
    s = fn(items)
    win = [int(s[i] > s[i + n]) for i in range(n)]
    R['rms'][label] = dict(chosen=s[:n], rejected=s[n:], correct=win)
    by = {sec: sum(w for w, p in zip(win, pairs) if p['section'] == sec) / sum(p['section'] == sec for p in pairs) for sec in SECTIONS}
    longer = [w for w, p in zip(win, pairs) if len(p['chosen']) > len(p['rejected'])]
    shorter = [w for w, p in zip(win, pairs) if len(p['chosen']) <= len(p['rejected'])]
    picks_longer = sum((s[i] > s[i + n]) == (len(pairs[i]['chosen']) > len(pairs[i]['rejected'])) for i in range(n)) / n
    R['rms'][label].update(by_section=by, overall=sum(win) / n, acc_chosen_longer=sum(longer) / len(longer),
                           acc_chosen_shorter=sum(shorter) / len(shorter), picks_longer=picks_longer)
    log(f'{label:21s} overall {sum(win) / n:.3f} | ' + '  '.join(f'{k} {v:.2f}' for k, v in by.items())
        + f' | acc when chosen is longer {sum(longer) / len(longer):.2f}, shorter {sum(shorter) / len(shorter):.2f}'
        + f' | picks the longer answer {picks_longer:.2f}')
save('ch3_rewardbench', R)
