"""Chapter 8: RLHF with PPO, from scratch, on Qwen2.5-0.5B-Instruct with the Chapter 7 reward model.

One PPO iteration:
  1. rollout   sample one reply (temperature 0.7, at most 128 tokens) for each of 32 prompts with the current policy
  2. score     reward model score at the last token (normalised; -1 if the reply never ended: the "EOS trick"),
               log-probs of the policy ("old") and of the reference, values from the value model
  3. rewards   per-token reward = -beta * (log pi_old - log pi_ref), plus the score at the last token
  4. GAE       advantages and returns with gamma = 1, lambda = 0.95; advantages whitened over the batch
  5. update    2 epochs over the batch in 2 minibatches of 16: clipped policy loss for the policy's LoRA adapter,
               clipped value loss for the value model
Usage: python ch8_ppo.py --name beta005 --beta 0.05 [--steps 100]"""
import argparse, json, math, os, random, time
import numpy as np
import torch
import torch.nn.functional as F
from common import Log, save, RESULTS
from ch7_common import load_rm, score_ids, tok, DEV, PAD
from ch8_common import load_policy, generate, batch_tensors, token_logprobs, rm_texts, load_prompts, CKPT

ap = argparse.ArgumentParser()
ap.add_argument('--name', required=True)
ap.add_argument('--beta', type=float, default=0.05)      # KL penalty coefficient
ap.add_argument('--steps', type=int, default=20)
ap.add_argument('--batch', type=int, default=32)         # prompts (= replies) per PPO iteration
ap.add_argument('--minibatches', type=int, default=2)
ap.add_argument('--epochs', type=int, default=2)         # PPO epochs over each batch
ap.add_argument('--micro', type=int, default=16)         # sequences per forward pass (gradient accumulation)
ap.add_argument('--lr', type=float, default=3e-5)        # policy LoRA learning rate
ap.add_argument('--vlr', type=float, default=3e-5)       # value model learning rate
ap.add_argument('--clip', type=float, default=0.2)       # epsilon of the clipped objective
ap.add_argument('--vclip', type=float, default=0.2)      # value clipping range
ap.add_argument('--lam', type=float, default=0.95)
ap.add_argument('--gamma', type=float, default=1.0)
ap.add_argument('--temp', type=float, default=0.7)
ap.add_argument('--max_new', type=int, default=128)
ap.add_argument('--eos_penalty', type=float, default=-1.0)   # normalised score given to replies that never ended
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--save_every', type=int, default=10)
a = ap.parse_args()

log = Log(f'ch8_ppo_{a.name}')
random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
P = load_prompts()
train_prompts = P['train']
log(f'run {a.name}: {vars(a)}')
log(f'{len(train_prompts)} training prompts')

# ------------------------------------------------------------------ the four models
policy = load_policy(seed=a.seed)                     # reference = the same model with the adapter disabled
rm = load_rm()                                         # frozen reward model
value = load_rm()                                      # value model: starts as a copy of the reward model ...
for n, p in value.named_parameters():                  # ... whose LoRA weights and score head train
    p.requires_grad = ('lora_' in n) or ('score' in n and 'modules_to_save' in n)
pol_params = [p for p in policy.parameters() if p.requires_grad]
val_params = [p for p in value.parameters() if p.requires_grad]
opt_p = torch.optim.AdamW(pol_params, lr=a.lr, eps=1e-5, weight_decay=0.0)
opt_v = torch.optim.AdamW(val_params, lr=a.vlr, eps=1e-5, weight_decay=0.0)
log(f'trainable: policy {sum(p.numel() for p in pol_params):,}  value {sum(p.numel() for p in val_params):,}')

# ------------------------------------------------------------------ reward normalisation (shared by all runs)
CAL = os.path.join(RESULTS, 'ch8_calib.json')
if not os.path.exists(CAL):
    rng = random.Random(123)
    cal_prompts = rng.sample(train_prompts, 192)
    torch.manual_seed(123)
    cal = []
    for k in range(0, len(cal_prompts), 48):
        cal += generate(policy, cal_prompts[k:k + 48], a.max_new, a.temp)
    raw = np.array(score_ids(rm, rm_texts(cal)))
    json.dump(dict(mean=float(raw.mean()), sd=float(raw.std()), n=len(raw),
                   finished=float(np.mean([s['finished'] for s in cal])),
                   mean_len=float(np.mean([len(s['r_ids']) for s in cal]))), open(CAL, 'w'), indent=1)
cal = json.load(open(CAL))
MU, SD = cal['mean'], cal['sd']
log(f'reward normalisation from {cal["n"]} replies of the starting model: raw mean {MU:.3f} sd {SD:.3f} '
    f'(finished {cal["finished"]:.2f}, mean length {cal["mean_len"]:.1f} tokens)')
torch.manual_seed(a.seed)


def values_of(ids, att):
    """Value of every state: the value model's score head read at every position (not just the last token),
    put on the same normalised scale as the rewards."""
    inner = value.base_model.model                     # Qwen2ForSequenceClassification with LoRA layers inside
    h = inner.model(input_ids=ids, attention_mask=att).last_hidden_state
    v = inner.score(h)[..., 0].float()
    return ((v - MU) / SD)[:, :-1]                     # position t = the state before choosing token t+1


def masked_mean(x, m):
    return (x * m).sum() / m.sum()


def whiten(x, m, shift_mean=True):
    mu = masked_mean(x, m)
    var = masked_mean((x - mu) ** 2, m)
    w = (x - mu) * torch.rsqrt(var + 1e-8)
    return (w if shift_mean else w + mu) * m


def gae(rew, val, act, gamma, lam):
    """Advantages by the backward recursion of Chapter 6 (Section 6.10), on right-padded rows.
    act marks the reply's action positions; the value after the last action is 0 (the episode is over)."""
    T = rew.shape[1]
    adv = torch.zeros_like(rew)
    last = torch.zeros(rew.shape[0], device=rew.device)
    nxt_v = torch.zeros(rew.shape[0], device=rew.device)
    for t in reversed(range(T)):
        m = act[:, t]
        delta = rew[:, t] + gamma * nxt_v - val[:, t]
        last = (delta + gamma * lam * last) * m
        adv[:, t] = last
        nxt_v = torch.where(m > 0, val[:, t], nxt_v * 0)   # once inside the reply, carry this state's value back
    return adv, adv + val


order = []
hist, samples_log = [], {}
t_start = time.time()
for step in range(a.steps + 1):
    # ---------------------------------------------------------------- 1. rollout
    if len(order) < a.batch:
        order = random.sample(train_prompts, len(train_prompts)) + order   # reshuffle when the prompts run out
    prompts = [order.pop() for _ in range(a.batch)]
    t0 = time.time()
    S = generate(policy, prompts, a.max_new, a.temp)
    t_gen = time.time() - t0
    ids, att, act = batch_tensors(S)
    # ---------------------------------------------------------------- 2. score
    with torch.no_grad():
        policy.eval(); value.eval()
        old_lp, ref_lp, old_v, ent = [], [], [], []
        for k in range(0, len(S), a.micro):
            sl = slice(k, k + a.micro)
            lp, e = token_logprobs(policy, ids[sl], att[sl], a.temp, with_entropy=True)
            old_lp.append(lp); ent.append(e)
            with policy.disable_adapter():
                ref_lp.append(token_logprobs(policy, ids[sl], att[sl], a.temp))
            old_v.append(values_of(ids[sl], att[sl]))
        old_lp, ref_lp, old_v, ent = (torch.cat(x) for x in (old_lp, ref_lp, old_v, ent))
        raw = np.array(score_ids(rm, rm_texts(S)))
        fin = torch.tensor([s['finished'] for s in S], device=DEV)
        score = torch.tensor((raw - MU) / SD, device=DEV, dtype=torch.float32)
        score = torch.where(fin, score, torch.full_like(score, a.eos_penalty))   # the EOS trick
        # ------------------------------------------------------------ 3. per-token rewards
        log_ratio = (old_lp - ref_lp) * act                       # k1 estimate of the KL, token by token
        rew = -a.beta * log_ratio
        last_idx = (act * torch.arange(act.shape[1], device=DEV)).argmax(1)   # position of each reply's last action
        rew[torch.arange(len(S)), last_idx] += score
        # ------------------------------------------------------------ 4. advantages
        adv, ret = gae(rew, old_v * act, act, a.gamma, a.lam)
        adv = whiten(adv, act)
    # ---------------------------------------------------------------- stats of this batch (before the update)
    ntok = act.sum(1)
    st = dict(step=step, rm_raw=float(raw.mean()), score=float(score.mean()),
              score_finished=float(((raw - MU) / SD)[fin.cpu().numpy()].mean()) if fin.any() else None,
              kl=float(log_ratio.sum(1).mean()), rlhf_reward=float(rew.sum(1).mean()),
              length=float(ntok.mean()), finished=float(fin.float().mean()),
              entropy=float(masked_mean(ent, act)), t_gen=t_gen)
    if step == a.steps:                                            # final batch: measure only
        hist.append(st); break
    # ---------------------------------------------------------------- 5. PPO update
    policy.train(); value.train()                                  # (no dropout anywhere: LoRA dropout is 0)
    mb = a.batch // a.minibatches
    pg_l, v_l, cf, akl, vcf = [], [], [], [], []
    for epoch in range(a.epochs):
        perm = torch.randperm(a.batch)
        for j in range(a.minibatches):
            idx = perm[j * mb:(j + 1) * mb]
            opt_p.zero_grad(); opt_v.zero_grad()
            n_act = act[idx].sum()
            for k in range(0, mb, a.micro):
                ii = idx[k:k + a.micro]
                m = act[ii]
                new_lp = token_logprobs(policy, ids[ii], att[ii], a.temp)
                ratio = torch.exp(new_lp - old_lp[ii])
                A = adv[ii]
                pg1 = -A * ratio
                pg2 = -A * torch.clamp(ratio, 1 - a.clip, 1 + a.clip)
                pg_loss = (torch.max(pg1, pg2) * m).sum() / n_act        # token-level mean over the minibatch
                pg_loss.backward()
                v = values_of(ids[ii], att[ii])
                v_clip = old_v[ii] + torch.clamp(v - old_v[ii], -a.vclip, a.vclip)
                vl = torch.max((v - ret[ii]) ** 2, (v_clip - ret[ii]) ** 2)
                v_loss = 0.5 * (vl * m).sum() / n_act
                v_loss.backward()
                with torch.no_grad():
                    lr_ = new_lp - old_lp[ii]
                    pg_l.append(float(pg_loss) * n_act.item()); v_l.append(float(v_loss) * n_act.item())
                    cf.append(float((((ratio - 1).abs() > a.clip).float() * m).sum()))
                    akl.append(float((((ratio - 1) - lr_) * m).sum()))      # k3 estimate of KL(old || new)
                    vcf.append(float((((v - old_v[ii]).abs() > a.vclip).float() * m).sum()))
            torch.nn.utils.clip_grad_norm_(pol_params, 1.0)
            torch.nn.utils.clip_grad_norm_(val_params, 1.0)
            opt_p.step(); opt_v.step()
    N = act.sum().item() * a.epochs
    with torch.no_grad():
        ev = 1 - float(masked_mean((ret - old_v) ** 2, act) / (masked_mean((ret - masked_mean(ret, act)) ** 2, act) + 1e-8))
    st.update(pg_loss=sum(pg_l) / N, v_loss=sum(v_l) / N, clipfrac=sum(cf) / N, approx_kl=sum(akl) / N,
              vclipfrac=sum(vcf) / N, explained_var=ev, t_step=time.time() - t0)
    hist.append(st)
    log(f'step {step:3d}  score {st["score"]:+.3f}  rm_raw {st["rm_raw"]:+.3f}  kl {st["kl"]:6.2f}  len {st["length"]:5.1f}  '
        f'fin {st["finished"]:.2f}  ent {st["entropy"]:.3f}  clipfrac {st["clipfrac"]:.3f}  approx_kl {st["approx_kl"]:.4f}  '
        f'v_loss {st["v_loss"]:.3f}  ev {ev:+.2f}  ({st["t_step"]:.0f}s, gen {t_gen:.0f}s)')
    if step == 0 and DEV == 'mps':
        log(f'Apple GPU memory after the first iteration: {torch.mps.driver_allocated_memory() / 2**30:.1f} GB')
    if step % 25 == 0:
        samples_log[step] = [dict(prompt=s['prompt'][:300], reply=tok.decode(s['r_ids']), finished=s['finished'],
                                  score=float(sc)) for s, sc in zip(S[:4], score.tolist())]
    if step == 0:                                                  # one reply, token by token, for the chapter
        cand = [i for i in range(len(S)) if S[i]['finished'] and 8 <= ntok[i] <= 40]   # a short finished reply
        i0 = min(cand, key=lambda i: int(ntok[i])) if cand else 0
        L = int(ntok[i0]); pos = torch.nonzero(act[i0]).flatten().tolist()
        samples_log['example'] = dict(
            prompt=S[i0]['prompt'], reply=tok.decode(S[i0]['r_ids']), finished=S[i0]['finished'],
            raw=float(raw[i0]), score=float(score[i0]),
            tokens=[tok.decode([int(ids[i0, p + 1])]) for p in pos], old_lp=old_lp[i0, pos].tolist(),
            ref_lp=ref_lp[i0, pos].tolist(), values=old_v[i0, pos].tolist(), rewards=rew[i0, pos].tolist(),
            adv_raw=gae(rew[i0:i0 + 1], (old_v * act)[i0:i0 + 1], act[i0:i0 + 1], a.gamma, a.lam)[0][0, pos].tolist(),
            ret=ret[i0, pos].tolist())
    if (step + 1) % a.save_every == 0:
        policy.save_pretrained(os.path.join(CKPT, a.name, f'step{step + 1}'))
        save(f'ch8_ppo_{a.name}', dict(args=vars(a), calib=cal, hist=hist, samples=samples_log))

log(f'final batch (step {a.steps}): score {hist[-1]["score"]:+.3f}  kl {hist[-1]["kl"]:.2f}  len {hist[-1]["length"]:.1f}  '
    f'finished {hist[-1]["finished"]:.2f}')
log(f'total time {(time.time() - t_start) / 60:.1f} min')
save(f'ch8_ppo_{a.name}', dict(args=vars(a), calib=cal, hist=hist, samples=samples_log))
