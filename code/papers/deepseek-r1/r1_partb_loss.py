"""Background page, experiment (a): the next-token loss of a language model, by hand and with the library.

1. Pre-training loss on one sentence with Qwen2.5-0.5B (a base model): our own log-softmax and gather
   (Method 1) against torch.nn.functional.cross_entropy and the model's own loss (Method 2).
2. The supervised fine-tuning (SFT) loss on one prompt/response pair: the same loss, counted only on the
   response tokens (labels = -100 on the prompt), by hand and with cross_entropy(ignore_index=-100).
Output: results/r1_partb_loss_stdout.txt and results/r1_partb_loss.json."""
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import common

log = common.Log('r1_partb_loss')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
BASE = 'Qwen/Qwen2.5-0.5B'
SENT = 'The capital of France is Paris, and the capital of Italy is Rome.'


def next_token_loss(logits, ids):
    """Method 1: the next-token cross-entropy written out.
    logits: (T, V) scores at each position; ids: (T,) the tokens. Position t predicts token t + 1."""
    pred = logits[:-1]                                   # (T-1, V): the guesses for tokens 1..T-1
    target = ids[1:]                                     # (T-1,):   the tokens that really came next
    m = pred.max(dim=-1, keepdim=True).values            # subtract the max so exp() cannot overflow
    log_z = m.squeeze(-1) + torch.log(torch.exp(pred - m).sum(-1))   # (T-1,): log of the softmax denominator
    log_p = pred.gather(-1, target[:, None]).squeeze(-1) - log_z     # (T-1,): log p(right token)
    return -log_p, -log_p.mean()                         # per-token losses and their mean


def sft_loss(logits, labels):
    """Method 1 for SFT: the same loss, but only where labels != -100 (the response tokens)."""
    per_tok, _ = next_token_loss(logits, labels.clamp(min=0))
    keep = (labels[1:] != -100).float()                  # (T-1,): 1 on response tokens, 0 on prompt tokens
    return (per_tok * keep).sum() / keep.sum()


with common.heavy('partb'):
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32).to(DEV).eval()

    # ------------------------------------------------------------------ 1. pre-training loss
    ids = tok(SENT, return_tensors='pt').input_ids.to(DEV)          # (1, T)
    with torch.no_grad():
        out = model(ids, labels=ids)
    logits = out.logits[0].float()                                    # (T, V)
    T, V = logits.shape
    per_tok, ours = next_token_loss(logits, ids[0])
    lib = F.cross_entropy(logits[:-1], ids[0, 1:])                    # Method 2
    log('== 1. Next-token loss of Qwen2.5-0.5B (base model) on one sentence')
    log(f'  sentence: {SENT!r}')
    log(f'  tokens T = {T}, vocabulary V = {V:,}, logits shape ({T}, {V})')
    log('  next token        p(next)   -log p')
    probs = torch.exp(-per_tok)
    rows = []
    for t in range(T - 1):
        s = tok.decode(ids[0, t + 1])
        rows.append(dict(token=s, p=round(probs[t].item(), 4), loss=round(per_tok[t].item(), 3)))
        log(f'  {s!r:16}  {probs[t].item():7.4f}  {per_tok[t].item():7.3f}')
    log(f'  mean loss, ours (Method 1)              : {ours.item():.4f}')
    log(f'  mean loss, F.cross_entropy (Method 2)    : {lib.item():.4f}')
    log(f'  mean loss, model(ids, labels=ids).loss   : {out.loss.item():.4f}')
    log(f'  max |ours - library| per token: {(per_tok - F.cross_entropy(logits[:-1], ids[0, 1:], reduction="none")).abs().max().item():.1e}')
    log(f'  perplexity exp(loss) = {math.exp(ours.item()):.2f};  a uniform guess would give loss ln(V) = {math.log(V):.2f}')
    pre = dict(sentence=SENT, T=T, V=V, rows=rows, ours=ours.item(), lib=lib.item(), hf=out.loss.item())

    # ------------------------------------------------------------------ 2. SFT loss on a prompt/response pair
    prompt = 'Question: What is 12 times 3?\nAnswer:'
    resp = ' 12 times 3 is 36. The answer is 36.'
    p_ids = tok(prompt).input_ids
    r_ids = tok(resp).input_ids
    ids2 = torch.tensor([p_ids + r_ids], device=DEV)
    labels = ids2.clone()
    labels[0, :len(p_ids)] = -100                                     # do not learn to write the question
    with torch.no_grad():
        out2 = model(ids2, labels=labels)
    lg2 = out2.logits[0].float()
    ours2 = sft_loss(lg2, labels[0])
    lib2 = F.cross_entropy(lg2[:-1], labels[0, 1:], ignore_index=-100)   # Method 2
    all2 = F.cross_entropy(lg2[:-1], ids2[0, 1:])
    log('')
    log('== 2. SFT loss: the same loss, only on the response tokens')
    log(f'  prompt {len(p_ids)} tokens (label -100), response {len(r_ids)} tokens')
    log(f'  SFT loss, ours (Method 1)                       : {ours2.item():.4f}')
    log(f'  SFT loss, F.cross_entropy(ignore_index=-100)     : {lib2.item():.4f}')
    log(f'  SFT loss, model(ids, labels=labels).loss         : {out2.loss.item():.4f}')
    log(f'  loss on every token (prompt too), for comparison  : {all2.item():.4f}')
    sft = dict(prompt=prompt, response=resp, n_prompt=len(p_ids), n_resp=len(r_ids), ours=ours2.item(),
               lib=lib2.item(), hf=out2.loss.item(), all=all2.item())

common.save('r1_partb_loss', dict(pretrain=pre, sft=sft))
