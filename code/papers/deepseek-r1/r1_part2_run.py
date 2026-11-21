"""Part 2: one real GRPO step on a real model.
Qwen/Qwen2.5-0.5B-Instruct samples a group of G = 8 answers to one GSM8K question (temperature 1, at most 512 new
tokens). A check gives reward 1 when the whole expression in the last \\boxed{} (math-verify) equals the reference, else 0.
Sampling uses the plain softmax (repetition_penalty 1.0, no top-p/top-k), the same distribution the log-probs use. We then compute Equation 3
(advantages), the per-token log-probabilities, the ratios (all 1 at the first step), the k3 KL (0 at the first step)
and the loss of Equation 1, with our own functions and with TRL's code. One AdamW step follows, and we measure how
the log-probability of every answer moved.
We go through the GSM8K test set from the start and keep the first question whose group has both right and wrong
answers; the groups before it are printed too (they show the "no signal" case).
Settings: epsilon 0.2 (PPO's usual value) and beta 0.001 (the later, long version of the R1 paper). The step is taken
twice from the same weights: learning rate 3e-6 (the long version's value) and 1e-7 (a small step that is easy to read). Runs inside the heavy-job lock. Writes results/r1_part2_run_stdout.txt and results/r1_part2_run.json."""
import re, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from math_verify import parse, verify
from common import Log, save, heavy
from r1_part2_grpo import group_advantages, ratio, k3_kl, grpo_loss
from r1_part2_trl import trl_loss

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
G, MAX_NEW, EPS, BETA, LR, LR_SMALL, MAX_Q = 8, 512, 0.2, 0.001, 3e-6, 1e-7, 12
SAMPLE_PLAIN = dict(no_repeat_ngram_size=0, min_p=None, typical_p=1.0, epsilon_cutoff=0.0, eta_cutoff=0.0)
SYSTEM = 'Please reason step by step, and put your final answer within \\boxed{}.'
log = Log('r1_part2_run')
R = {}


def last_boxed(text):
    """The content of the last \\boxed{...}, with nested braces kept (a balanced-brace scan), or None."""
    i = text.rfind('\\boxed{')
    if i < 0:
        return None
    j, depth = i + len('\\boxed{'), 1
    for k in range(j, len(text)):
        depth += {'{': 1, '}': -1}.get(text[k], 0)
        if depth == 0:
            return text[j:k]
    return None                                                       # never closed: no answer


def final_answer(text):
    """The whole expression in the last \\boxed{...} if there is one, else the last number in the text."""
    boxed = last_boxed(text)
    cand = boxed if boxed is not None else (re.findall(r'-?[\d,]*\.?\d+', text) or [''])[-1]
    return cand.replace('\\$', '').replace('$', '').strip().rstrip('.')


def reward(text, gold):
    """1.0 if the whole final expression equals the reference answer (math-verify compares the expressions,
    so 3/2 is 1.5, not 3), else 0.0."""
    ans = final_answer(text)
    if not ans:
        return 0.0
    try:
        return float(verify(parse(gold), parse(f'${ans}$')))
    except Exception:
        return 0.0


def per_token_logps(model, ids, P):
    """log pi(o_t | q, o_<t) for every answer token. ids: (B, P + T). Returns (B, T)."""
    logits = model(ids).logits[:, P - 1:-1].float()                # (B, T, V): the logits that predict each answer token
    return torch.log_softmax(logits, -1).gather(-1, ids[:, P:].unsqueeze(-1)).squeeze(-1)   # (B, T)


with heavy('part2'):
    t0 = time.time()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).to(dev)
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    log(f'model {MODEL} on {dev}, float32; G = {G}, temperature 1.0, max_new_tokens {MAX_NEW}')
    # The checkpoint's generation_config.json sets repetition_penalty 1.1 (and temperature 0.7, top_p 0.8, top_k 20).
    # Any of these would make the sampling policy differ from the plain softmax that per_token_logps computes, so the
    # ratio pi_theta / pi_theta_old would not start at the true value. We switch every one of them off.
    log(f'checkpoint defaults: repetition_penalty {model.generation_config.repetition_penalty}, '
        f'temperature {model.generation_config.temperature}, top_p {model.generation_config.top_p}, '
        f'top_k {model.generation_config.top_k}')
    log('sampling with: temperature 1.0, top_p 1.0, top_k 0, repetition_penalty 1.0 (plain softmax, the same '
        'distribution the log-probs use)')
    model.eval()                                                      # no dropout while sampling or scoring
    log('')
    log('== 1. Sample a group of 8 answers per question, reward 1 if the final answer is right')
    for qi in range(MAX_Q):
        q, gold = ds[qi]['question'], ds[qi]['answer'].split('####')[-1].strip().replace(',', '')
        msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, return_tensors='pt').input_ids.to(dev)
        P = p_ids.shape[1]
        torch.manual_seed(qi)
        with torch.no_grad():
            out = model.generate(p_ids, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=MAX_NEW,
                                 repetition_penalty=1.0, num_return_sequences=G, pad_token_id=tok.pad_token_id,
                                 **SAMPLE_PLAIN)
        comp = out[:, P:]
        texts = tok.batch_decode(comp, skip_special_tokens=True)
        r = torch.tensor([reward(t, gold) for t in texts])
        A = group_advantages(r, G)
        log(f'question {qi}: gold {gold}; rewards {[int(v) for v in r.tolist()]}; '
            f'advantages [{", ".join(f"{v:+.2f}" for v in A.tolist())}]')
        if 0 < r.sum() < G:
            break
    log('')
    log(f'== 2. The group we train on: GSM8K test question {qi}')
    log(f'question: {q}')
    log(f'reference answer: {gold}')
    # the mask: 1 for every answer token up to and including the first end token, 0 for the padding after it
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids('<|im_end|>'), tok.pad_token_id}
    mask = torch.ones_like(comp, dtype=torch.float32)
    for i in range(G):
        ends = [t for t in range(comp.shape[1]) if comp[i, t].item() in eos]
        if ends:
            mask[i, ends[0] + 1:] = 0
    lens = mask.sum(1).int().tolist()
    for i in range(G):
        log(f'answer {i}: {lens[i]:3d} tokens, final answer {final_answer(texts[i]):>8s}, reward {int(r[i])}, A {A[i]:+.3f}')
    best = int((r == 1).nonzero()[0])                               # the first right answer
    worst = int((r == 0).nonzero()[0])                              # the first wrong answer
    log('')
    log(f'answer {best} (right), last 220 characters: ...{texts[best][-220:]!r}')
    log(f'answer {worst} (wrong), last 220 characters: ...{texts[worst][-220:]!r}')

    log('')
    log('== 3. Per-token log-probs, ratios and KL at the first step')
    ids = out                                                        # (G, P + T): prompt + answer
    logp = per_token_logps(model, ids, P)                            # (G, T), carries the gradient
    logp_old = logp.detach()                                         # the sampling policy is this same model
    with torch.no_grad():
        logp_ref = per_token_logps(model, ids, P)                    # the reference: at step 1 also the same weights
    m = mask.to(dev)
    log(f'shapes: ids {tuple(ids.shape)}, per-token log-probs {tuple(logp.shape)}, mask {tuple(m.shape)}')
    toks = tok.convert_ids_to_tokens(comp[best, :10])
    log(f'answer {best}, first 10 tokens and log pi_theta:')
    log('  ' + '  '.join(f'{t.replace("Ġ", "_").replace("Ċ", "/n")}:{v:.2f}' for t, v in zip(toks, logp_old[best, :10].tolist())))
    seq_lp = (logp_old * m).sum(1)
    log(f'log pi_theta(o_i | q) of each whole answer (sum over its tokens): '
        f'[{", ".join(f"{v:.1f}" for v in seq_lp.tolist())}]')
    rho = ratio(logp, logp_old)
    log(f'ratios pi_theta / pi_theta_old on real tokens: min {rho[m.bool()].min().item():.4f}, '
        f'max {rho[m.bool()].max().item():.4f}')
    kl = k3_kl(logp.detach(), logp_ref)
    log(f'max |log pi_theta - log pi_ref| {(logp.detach() - logp_ref)[m.bool()].abs().max().item():.2e}; '
        f'k3 KL mean over real tokens {kl[m.bool()].mean().item():.2e} (0 up to float rounding)')
    A_d = A.to(dev)
    loss = grpo_loss(logp, logp_old, logp_ref, A_d, m, epsilon=EPS, beta=BETA)
    loss_trl, _ = trl_loss(logp.detach().cpu(), logp_old.cpu(), logp_ref.cpu(), A, mask.cpu(), epsilon=EPS, beta=BETA)
    log(f'loss, ours {loss.item():+.8f}; TRL {loss_trl.item():+.8f}; |difference| {abs(loss.item() - loss_trl.item()):.1e}')
    log(f'mean of the advantages {A.mean().item():+.1e}: at ratio 1 the loss is -(1/G) sum A_i = 0, '
        f'but its gradient is not 0')

    log('')
    log(f'agree to 1e-6: {abs(loss.item() - loss_trl.item()) < 1e-6}')
    log('')
    log('== 4. One gradient step (AdamW) and its effect on the log-probability of each answer')
    loss.backward()
    gn = torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None))
    log(f'gradient norm {gn.item():.3f}')
    start = {k: v.detach().clone() for k, v in model.state_dict().items()}   # the weights before the step
    grads = [p.grad.detach().clone() for p in model.parameters()]
    results = {}
    for lr in (LR, LR_SMALL):                                         # the long version's value, then a small one
        model.load_state_dict(start)
        for p_, g_ in zip(model.parameters(), grads):
            p_.grad = g_.clone()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        opt.step()
        with torch.no_grad():
            lp = per_token_logps(model, ids, P)
        results[lr] = lp
        sn = (lp * m).sum(1)
        log(f'-- learning rate {lr}')
        log('answer  reward      A     log pi before   log pi after   change')
        rows = []
        for i in range(G):
            d = (sn[i] - seq_lp[i]).item()
            rows.append(dict(i=i, reward=r[i].item(), A=A[i].item(), before=seq_lp[i].item(), after=sn[i].item(), tokens=lens[i]))
            log(f'{i:5d}   {int(r[i]):5d}   {A[i]:+.3f}   {seq_lp[i].item():12.2f}   {sn[i].item():12.2f}   {d:+.3f}')
        up = [x for x in rows if x['A'] > 0]
        dn = [x for x in rows if x['A'] < 0]
        log(f'right answers: mean change {sum(x["after"] - x["before"] for x in up) / len(up):+.3f}; '
            f'wrong answers: mean change {sum(x["after"] - x["before"] for x in dn) / len(dn):+.3f}')
        log(f'answer {best} (right): probability of the whole answer x{torch.exp(sn[best] - seq_lp[best]).item():.4g}')
        R[f'rows_{lr}'] = rows
    rows = R[f'rows_{LR_SMALL}']
    logp_new = results[LR_SMALL]
    log('')
    log(f'== 5. The same batch after the step with learning rate {LR_SMALL}: ratios move away from 1')
    rho2 = ratio(logp_new, logp_old)
    rr = rho2[m.bool()]
    log(f'per-token ratios: min {rr.min().item():.4f}, max {rr.max().item():.4f}, '
        f'outside [0.8, 1.2]: {((rr < 0.8) | (rr > 1.2)).float().mean().item() * 100:.2f}% of {rr.numel()} tokens')
    log(f'answer {best}: whole-answer ratio exp(sum of log ratios) = {torch.exp(((logp_new - logp_old) * m)[best].sum()).item():.3f}')
    kl2 = k3_kl(logp_new, logp_ref)
    log(f'k3 KL to the reference, mean over real tokens: {kl2[m.bool()].mean().item():.2e}')
    l1 = grpo_loss(logp_new, logp_old, logp_ref, A_d, m, epsilon=EPS, beta=BETA)
    l2, met = trl_loss(logp_new.cpu(), logp_old.cpu(), logp_ref.cpu(), A, mask.cpu(), epsilon=EPS, beta=BETA)
    log(f'loss at the second step, ours {l1.item():+.8f}; TRL {l2.item():+.8f}; |difference| {abs(l1.item() - l2.item()):.1e}')
    log(f'share of tokens in the flat (clipped) region, TRL metric: {met["clip_ratio/region_mean"][0] * 100:.2f}%')
    log(f'(the loss went from {loss.item():+.6f} to {l1.item():+.6f})')
    log(f'time {time.time() - t0:.0f} s')
    R.update(question_index=qi, question=q, gold=gold, rewards=r.tolist(), advantages=A.tolist(), lengths=lens,
             rows=rows, best=best, worst=worst, loss1=loss.item(), loss2=l1.item())
    save('r1_part2_run', R)
