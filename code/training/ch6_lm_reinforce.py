"""Chapter 6: REINFORCE on a real language model. GPT-2 (124M) continues the first 8 tokens of IMDB movie reviews with 24 new
tokens; the reward is a sentiment classifier's log-odds that the whole text is positive. A frozen copy of GPT-2 is the reference
model for the per-token KL penalty. Usage: python ch6_lm_reinforce.py <name> <beta> [steps] [baseline: loo|none]
Writes results/ch6_lm_<name>.json (curves, samples, held-out evaluation) and results/ch6_lm_<name>_stdout.txt."""
import sys, json, random, re
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from common import Log, save, timer

NAME, BETA = sys.argv[1], float(sys.argv[2])
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 200
BASELINE = sys.argv[4] if len(sys.argv) > 4 else 'loo'
import os
P_LEN, R_LEN, N_PROMPTS, K = 8, 24, 16, 4
LR = float(os.environ.get('LR', '2e-5'))    # prompt tokens, reply tokens, prompts per step, samples per prompt
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
torch.manual_seed(0); random.seed(0)
log = Log(f'ch6_lm_{NAME}')
clock = timer()

tok = AutoTokenizer.from_pretrained('gpt2')
EOS = tok.eos_token_id
policy = AutoModelForCausalLM.from_pretrained('gpt2').to(dev)
ref = AutoModelForCausalLM.from_pretrained('gpt2').to(dev).eval()
for p in ref.parameters():
    p.requires_grad_(False)
ctok = AutoTokenizer.from_pretrained('lvwerra/distilbert-imdb')
clf = AutoModelForSequenceClassification.from_pretrained('lvwerra/distilbert-imdb').to(dev).eval()
POS = clf.config.label2id.get('POSITIVE', 1)


def prompts(split, n, seed):
    d = load_dataset('stanfordnlp/imdb', split=split).shuffle(seed=seed)
    out = []
    for ex in d:
        ids = tok(re.sub(r'<br\s*/?>', ' ', ex['text']))['input_ids'][:P_LEN]
        if len(ids) == P_LEN:
            out.append(ids)
        if len(out) == n:
            return out


TRAIN, EVAL = prompts('train', 4000, 0), prompts('test', 64, 1)


@torch.no_grad()
def sample(model, prompt_ids):
    """Plain ancestral sampling at temperature 1 with the end-of-text token banned, so every reply has exactly R_LEN tokens."""
    x = torch.tensor(prompt_ids, device=dev)
    out = model(x, use_cache=True)
    past, logits, new = out.past_key_values, out.logits[:, -1], []
    for _ in range(R_LEN):
        logits[:, EOS] = -float('inf')
        nxt = torch.multinomial(F.softmax(logits.float(), -1), 1)
        new.append(nxt)
        out = model(nxt, past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1]
    return torch.cat([x, torch.cat(new, 1)], 1)


def token_logits(model, seq):
    """Logits that predicted each reply token (the token shift of Chapter 1), with end-of-text banned as in sampling."""
    lg = model(seq).logits[:, P_LEN - 1:-1].float()
    lg[..., EOS] = -float('inf')
    return lg


@torch.no_grad()
def reward(seq):
    texts = tok.batch_decode(seq, skip_special_tokens=True)
    enc = ctok(texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(dev)
    lg = clf(**enc).logits.float()
    return lg[:, POS] - lg[:, 1 - POS], texts             # log-odds of "positive"


def distinct(texts, n):
    grams = [tuple(t.split()[i:i + n]) for t in texts for i in range(max(0, len(t.split()) - n + 1))]
    return len(set(grams)) / max(1, len(grams))


@torch.no_grad()
def stats(seq):
    """Reward, exact KL to the reference per reply (summed over tokens), entropy and reference log-prob per token."""
    lp_all = F.log_softmax(token_logits(policy, seq), -1)
    lq_all = F.log_softmax(token_logits(ref, seq), -1)
    tgt = seq[:, P_LEN:, None]
    p_all = lp_all.exp()
    kl = (p_all * (lp_all - lq_all)).nan_to_num().sum(-1)            # exact KL(pi || ref) at each position
    ent = -(p_all * lp_all).nan_to_num().sum(-1)
    lp, lq = lp_all.gather(-1, tgt)[..., 0], lq_all.gather(-1, tgt)[..., 0]
    R, texts = reward(seq)
    return dict(R=R, kl_exact=kl.sum(1), k1=(lp - lq).sum(1), ent=ent.mean(1), ref_lp=lq.mean(1), texts=texts, lp=lp, lq=lq)


@torch.no_grad()
def evaluate(tag):
    seq = torch.cat([sample(policy, [p] * K) for p in EVAL])
    s = stats(seq)
    replies = tok.batch_decode(seq[:, P_LEN:], skip_special_tokens=True)
    res = dict(R=s['R'].mean().item(), p_pos=torch.sigmoid(s['R']).mean().item(), kl=s['kl_exact'].mean().item(),
               ent=s['ent'].mean().item(), ref_ppl=torch.exp(-s['ref_lp']).mean().item(),
               d1=distinct(replies, 1), d2=distinct(replies, 2))
    log(f'[eval {tag}] {len(EVAL)} held-out prompts x {K}: reward {res["R"]:+.2f}  P(positive) {res["p_pos"]:.3f}  '
        f'KL {res["kl"]:.2f} nats/reply  entropy {res["ent"]:.2f}  ref-perplexity {res["ref_ppl"]:.1f}  '
        f'distinct-1 {res["d1"]:.3f}  distinct-2 {res["d2"]:.3f}')
    res['samples'] = [s['texts'][i] for i in range(0, 4 * K * 3, K)][:12]
    res['sample_R'] = [s['R'][i].item() for i in range(0, 4 * K * 3, K)][:12]
    return res, seq, s


opt = torch.optim.Adam(policy.parameters(), lr=LR)
log(f'run {NAME}: beta = {BETA}, baseline = {BASELINE}, {STEPS} steps x {N_PROMPTS} prompts x {K} samples, '
    f'{P_LEN}+{R_LEN} tokens, Adam lr {LR}, device {dev}')
OUT = dict(name=NAME, beta=BETA, baseline=BASELINE, steps=STEPS, lr=LR, curve=[], samples={})
OUT['eval_start'], _, _ = evaluate('start')
for step in range(STEPS + 1):
    batch = random.sample(TRAIN, N_PROMPTS)
    seq = sample(policy, [p for p in batch for _ in range(K)])
    lg = token_logits(policy, seq)                                    # with gradient
    logp = F.log_softmax(lg, -1).gather(-1, seq[:, P_LEN:, None])[..., 0]
    with torch.no_grad():
        ref_logp = F.log_softmax(token_logits(ref, seq), -1).gather(-1, seq[:, P_LEN:, None])[..., 0]
        R, texts = reward(seq)
        # per-token rewards: a KL penalty at every token, the classifier score added at the last token
        kl_tok = logp.detach() - ref_logp
        r_tok = -BETA * kl_tok
        r_tok[:, -1] += R
        G = r_tok.flip(1).cumsum(1).flip(1)                           # reward-to-go G_t (gamma = 1)
        if BASELINE == 'loo':
            Gk = G.view(N_PROMPTS, K, R_LEN)
            b = (Gk.sum(1, keepdim=True) - Gk) / (K - 1)              # mean of the other samples of the same prompt
            A = (Gk - b).view(-1, R_LEN)
        else:
            A = G
    loss = -(A * logp).mean()
    opt.zero_grad()
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0).item()
    opt.step()
    replies = tok.batch_decode(seq[:, P_LEN:], skip_special_tokens=True)
    row = dict(step=step, R=R.mean().item(), p_pos=torch.sigmoid(R).mean().item(), k1=kl_tok.sum(1).mean().item(),
               d2=distinct(replies, 2), gn=gn, loss=loss.item(), adv_sd=A.std().item())
    OUT['curve'].append(row)
    if step % 10 == 0:
        log(f'step {step:4d}  reward {row["R"]:+.2f}  P(pos) {row["p_pos"]:.3f}  KL(k1) {row["k1"]:6.2f}  distinct-2 {row["d2"]:.3f}  '
            f'grad-norm {gn:6.2f}  {clock():5.0f}s')
    if step % 50 == 0:
        OUT['samples'][step] = [texts[i] for i in range(0, 3 * K, K)]
        for t in OUT['samples'][step]:
            log('    ' + repr(t)[:160])
        save(f'ch6_lm_{NAME}', OUT)
OUT['eval_end'], seq, s = evaluate('end')

# a worked example: one reply, token by token
i = 0
toks = [tok.decode([t]) for t in seq[i, P_LEN:].tolist()]
lp, lq = s['lp'][i].tolist(), s['lq'][i].tolist()
Ri = s['R'][i].item()
rt = [-BETA * (a - b) for a, b in zip(lp, lq)]
rt[-1] += Ri
G = [sum(rt[j:]) for j in range(R_LEN)]
log(f'\nworked example (beta = {BETA}): prompt {tok.decode(seq[i, :P_LEN])!r}, classifier log-odds R = {Ri:+.3f}')
for j in range(R_LEN):
    log(f'  {j:2d} {toks[j]!r:14s} log pi {lp[j]:7.3f}  log ref {lq[j]:7.3f}  log-ratio {lp[j] - lq[j]:+7.3f}  r_t {rt[j]:+7.3f}  G_t {G[j]:+7.3f}')
log(f'  sum of log-ratios = {sum(lp) - sum(lq):.3f};  total reward = R - beta * sum = {Ri - BETA * (sum(lp) - sum(lq)):+.3f} = G_0')
OUT['worked'] = dict(prompt=tok.decode(seq[i, :P_LEN]), toks=toks, lp=lp, lq=lq, R=Ri, rt=rt, G=G)
OUT['minutes'] = clock() / 60
save(f'ch6_lm_{NAME}', OUT)
log(f'done in {clock() / 60:.1f} min')
