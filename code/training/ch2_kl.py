"""Chapter 2, experiment 2: the RLHF objective "reward minus beta times KL" (InstructGPT, eq. 2) on numbers.

Part A, a toy with 4 possible answers, small enough to solve exactly. The best policy for
    maximise  E_y~pi [ r(y) ]  -  beta * KL(pi || pi_ref)
is known in closed form: pi*(y) = pi_ref(y) * exp(r(y) / beta) / Z  (Ziegler et al. 2019, DPO paper eq. 4).
We print it for several beta, with the reward-model score and a separate "true quality" a person would give.

Part B, real models: how far is Qwen2.5-0.5B-Instruct from its starting point Qwen2.5-0.5B, measured the way the
RLHF penalty measures it: the log-probability ratio of sampled answers, summed over tokens."""
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save

log = Log('ch2_kl')
OUT = {}

# ---------------------------------------------------------------- Part A: the toy
log('== A. four possible answers to one prompt ==')
names = ['A: correct, helpful', 'B: correct, terse', 'C: vague', 'D: odd text the RM loves']
pi_ref = np.array([0.40, 0.35, 0.2499, 0.0001])   # how likely the SFT model is to write each one
r = np.array([1.0, 0.6, -0.5, 2.5])                # what the reward model says
true_q = np.array([1.0, 0.6, -0.5, -3.0])           # what a careful person would say: D is junk the RM over-rates
for n, p, rr, q in zip(names, pi_ref, r, true_q):
    log(f'{n:<26} pi_ref {p:.4f}   reward {rr:+.1f}   true quality {q:+.1f}')


def kl(p, q):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


log('\n beta | pi*(A)  pi*(B)  pi*(C)  pi*(D) | E[reward]  KL(nats)  objective | E[true quality]')
rowsA = []
for beta in [10, 2, 1, 0.5, 0.25, 0.1, 0.05]:
    w = pi_ref * np.exp(r / beta)
    pi = w / w.sum()
    er, k = float(pi @ r), kl(pi, pi_ref)
    rowsA.append(dict(beta=beta, pi=pi.round(4).tolist(), reward=er, kl=k, objective=er - beta * k, true=float(pi @ true_q)))
    log(f'{beta:>5} | ' + '  '.join(f'{x:.3f}' for x in pi) + f' |   {er:+.3f}    {k:.3f}     {er - beta * k:+.3f}   |   {pi @ true_q:+.3f}')
log(f'reference policy itself: E[reward] {pi_ref @ r:+.3f}, KL 0, E[true quality] {pi_ref @ true_q:+.3f}')
OUT['toy'] = dict(names=names, pi_ref=pi_ref.tolist(), r=r.tolist(), true=true_q.tolist(), rows=rowsA,
                  ref_reward=float(pi_ref @ r), ref_true=float(pi_ref @ true_q))

# the per-sample quantity PPO actually optimises: r(x, y) - beta * log(pi(y|x) / pi_ref(y|x))
log('\n== per-sample penalised reward, beta = 0.5, policy = pi* for beta 0.5 ==')
beta = 0.5
pi = pi_ref * np.exp(r / beta); pi /= pi.sum()
per = []
for n, p, q, rr in zip(names, pi, pi_ref, r):
    lr = np.log(p / q)
    per.append(dict(name=n, pi=float(p), ref=float(q), logratio=float(lr), r=float(rr), total=float(rr - beta * lr)))
    log(f'{n:<26} log(pi/pi_ref) = log({p:.4f}/{q:.4f}) = {lr:+.3f}   r - beta*logratio = {rr:+.2f} - {beta}*({lr:+.3f}) = {rr - beta * lr:+.3f}')
log('(at the optimum every answer gets the same penalised reward: that number is beta * log Z)')
OUT['per_sample'] = dict(beta=beta, rows=per)

# ---------------------------------------------------------------- Part B: real models
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
BASE, INST = 'Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct'
tok = AutoTokenizer.from_pretrained(INST)
pol = AutoModelForCausalLM.from_pretrained(INST, dtype=torch.float32).to(DEV).eval()
ref = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(DEV).eval()


@torch.no_grad()
def token_logps(model, ids, start):
    """log pi(token_t | tokens before t) for every token from position `start` on."""
    logits = model(torch.tensor([ids], device=DEV)).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    tgt = torch.tensor(ids[1:], device=DEV)
    return lp[torch.arange(len(ids) - 1), tgt][start - 1:].cpu().numpy()


PROMPT = 'Explain in two sentences why the sky is blue.'
prompt_ids = tok.apply_chat_template([{'role': 'user', 'content': PROMPT}], add_generation_prompt=True, tokenize=True, return_dict=False)
log(f'\n== B. real models: policy {INST}, reference {BASE} ==')
log(f'prompt: {PROMPT!r} ({len(prompt_ids)} tokens with the chat template)')
torch.manual_seed(0)
seqs = []
gen = pol.generate(torch.tensor([prompt_ids] * 8, device=DEV), do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                   max_new_tokens=80, pad_token_id=tok.convert_tokens_to_ids('<|endoftext|>'),
                   eos_token_id=tok.convert_tokens_to_ids('<|im_end|>'))
end_id = tok.convert_tokens_to_ids('<|im_end|>')
rowsB = []
for g in gen:
    new = g[len(prompt_ids):].tolist()
    if end_id in new:
        new = new[:new.index(end_id) + 1]
    ids = prompt_ids + new
    lp_pol = token_logps(pol, ids, len(prompt_ids))
    lp_ref = token_logps(ref, ids, len(prompt_ids))
    lr = lp_pol - lp_ref
    rowsB.append(dict(text=tok.decode(new, skip_special_tokens=True), n=len(new), logp_pol=float(lp_pol.sum()),
                      logp_ref=float(lp_ref.sum()), logratio=float(lr.sum()), per_token=lr.round(3).tolist()[:12],
                      tokens=[tok.decode([t]) for t in new[:12]]))
for i, row in enumerate(rowsB):
    log(f'sample {i}: {row["n"]:>2} tokens  log pi = {row["logp_pol"]:8.2f}  log pi_ref = {row["logp_ref"]:8.2f}  '
        f'log-ratio = {row["logratio"]:6.2f}   "{row["text"][:60]}..."')
mean_lr = float(np.mean([x['logratio'] for x in rowsB]))
log(f'average log-ratio over 8 samples (a Monte Carlo estimate of KL(pi || pi_ref) for this prompt): {mean_lr:.2f} nats')
log(f'with beta = 0.02 (InstructGPT): penalty = 0.02 * {mean_lr:.2f} = {0.02 * mean_lr:.3f} reward points per answer')
first = rowsB[0]
log('\nfirst sample, token by token (log pi - log pi_ref):')
for t, v in zip(first['tokens'], first['per_token']):
    log(f'  {t!r:<14} {v:+.3f}')
OUT['real'] = dict(prompt=PROMPT, rows=rowsB, mean_logratio=mean_lr, beta=0.02)
save('ch2_kl', OUT)
