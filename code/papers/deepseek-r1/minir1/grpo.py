"""GRPO, §2.2.1 of the paper: sample a group of answers per question, score them with rules, compare each answer with
its own group (Eq. 3), and push the policy with a clipped ratio (Eq. 1) and a KL leash to a frozen reference (Eq. 2).
Shapes: B = number of completions (questions x G), P = prompt tokens, T = completion tokens, V = vocabulary."""
import torch
from .template import full_completion


@torch.no_grad()
def sample_group(model, tok, prompts, G, max_new_tokens, temperature=1.0):
    """Sample G completions for each prompt from the current policy (the 'old policy' of Eq. 1)."""
    texts = [p for p in prompts for _ in range(G)]               # B = len(prompts) * G, groups side by side
    enc = tok(texts, return_tensors='pt', padding=True, padding_side='left').to(model.device)
    out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=0,
                         repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                         stop_strings=['</answer>'], tokenizer=tok, pad_token_id=tok.pad_token_id)
    prompt_ids, completion_ids = enc.input_ids, out[:, enc.input_ids.size(1):]          # (B, P), (B, T)
    lengths = completion_lengths(completion_ids.tolist(), tok, end_token_ids(model, tok))
    T = completion_ids.size(1)
    completion_mask = (torch.arange(T)[None, :] < torch.tensor(lengths)[:, None]).long().to(model.device)  # (B, T)
    texts = [full_completion(tok.decode(ids[:n], skip_special_tokens=True))            # only the real tokens
             for ids, n in zip(completion_ids.tolist(), lengths)]
    return dict(prompt_ids=prompt_ids, prompt_mask=enc.attention_mask,
                completion_ids=completion_ids, completion_mask=completion_mask, texts=texts)


def end_token_ids(model, tok):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids])
    return ids | {tok.eos_token_id}


def completion_lengths(rows, tok, end_ids, stop='</answer>'):
    """How many tokens of each row the model really wrote. generate() fills a finished row with padding, and for Qwen
    the padding token is also an end-of-text token. So: a row that stopped on the stop string ends with the token that
    completes it (the padding after it is not counted); a row that stopped on an end token keeps that token (it was
    sampled); a row that hit the length limit keeps everything."""
    out = []
    for ids in rows:
        k = next((i for i, t in enumerate(ids) if t in end_ids or t == tok.pad_token_id), len(ids))   # first end/pad
        n = k
        if stop in tok.decode(ids[:k]):
            while n > 1 and stop in tok.decode(ids[:n - 1]):
                n -= 1                                       # the shortest prefix that holds the stop string
        elif k < len(ids):
            n = k + 1                                        # a real end token, chosen by the model: it counts
        out.append(n)
    return out


def per_token_logps(model, batch, temperature=1.0):
    """log pi(o_t | q, o_<t) for every completion token: the numbers that go into the ratio of Eq. 1."""
    ids = torch.cat([batch['prompt_ids'], batch['completion_ids']], 1)                  # (B, P + T)
    mask = torch.cat([batch['prompt_mask'], batch['completion_mask']], 1)               # (B, P + T)
    T = batch['completion_ids'].size(1)
    # The logits at position t predict token t + 1, so keep the T + 1 last positions and drop the very last one.
    logits = model(input_ids=ids, attention_mask=mask, logits_to_keep=T + 1).logits[:, :-1].float()   # (B, T, V)
    logits = logits / temperature
    chosen = logits.gather(-1, batch['completion_ids'].unsqueeze(-1)).squeeze(-1)      # (B, T): score of the token
    return chosen - torch.logsumexp(logits, -1)                                         # (B, T): log softmax


def group_advantages(rewards, G):
    """Eq. 3: A_i = (r_i - mean(r_1..r_G)) / std(r_1..r_G), each answer against its own group."""
    r = rewards.view(-1, G)                                                             # (questions, G)
    mean, std = r.mean(1, keepdim=True), r.std(1, keepdim=True)                         # std with n - 1, as TRL
    return ((r - mean) / (std + 1e-4)).view(-1)                                         # (B,): 0 when all equal


def grpo_loss(logps, old_logps, ref_logps, advantages, mask, eps=0.2, beta=0.04):
    """Eq. 1 with Eq. 2, per token, as a loss to minimise (the paper maximises J, so the sign flips)."""
    ratio = torch.exp(logps - old_logps)                         # pi_theta / pi_theta_old, (B, T)
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps)               # clip(ratio, 1 - eps, 1 + eps)
    A = advantages.unsqueeze(1)                                  # (B, 1): one advantage for every token of answer i
    surrogate = torch.min(ratio * A, clipped * A)                # the pessimistic one of the two
    log_r = ref_logps - logps                                    # log(pi_ref / pi_theta)
    kl = torch.exp(log_r) - log_r - 1                            # Eq. 2, always >= 0
    per_token = -(surrogate - beta * kl)                         # (B, T)
    m = mask.float()
    per_answer = (per_token * m).sum(1) / m.sum(1).clamp(min=1)  # 1/|o_i| sum over the tokens of answer i
    loss = per_answer.mean()                                     # 1/G sum over answers (and over questions)
    with torch.no_grad():   # sums and a token count, so the loop can take one token-weighted mean over a whole step
        # A token is clipped when the min of Eq. 1 picks the flat branch: ratio < 1 - eps with A < 0, or
        # ratio > 1 + eps with A > 0 (TRL's clip_ratio/region_mean counts the same tokens).
        is_clipped = ((ratio < 1 - eps) & (A < 0)) | ((ratio > 1 + eps) & (A > 0))
        stats = dict(kl_sum=(kl * m).sum().item(), clipped_sum=(is_clipped.float() * m).sum().item(),
                     tokens=m.sum().item())
    return loss, stats
