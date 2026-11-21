"""Held-out check: greedy answers to GSM8K test questions, scored with the same two rules as in training."""
import torch
from .template import make_prompt, full_completion
from .rewards import accuracy_reward, format_reward
from .grpo import end_token_ids, completion_lengths


@torch.no_grad()
def evaluate(model, tok, problems, max_new_tokens=300, batch=32):
    model.eval()
    end_ids = end_token_ids(model, tok)
    out = []
    for i in range(0, len(problems), batch):
        chunk = problems[i:i + batch]
        enc = tok([make_prompt(p['question']) for p in chunk], return_tensors='pt', padding=True,
                  padding_side='left').to(model.device)
        # Greedy, with no repetition penalty: the Instruct model's own generation config would add one of 1.1.
        gen = model.generate(**enc, do_sample=False, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                             stop_strings=['</answer>'], tokenizer=tok, pad_token_id=tok.pad_token_id)
        rows = gen[:, enc.input_ids.size(1):].tolist()
        for p, ids, n in zip(chunk, rows, completion_lengths(rows, tok, end_ids)):   # n: tokens really written
            text = full_completion(tok.decode(ids[:n], skip_special_tokens=True))
            out.append(dict(question=p['question'], gold=p['gold'], text=text, length=n,
                            acc=accuracy_reward(text, p['gold']), fmt=format_reward(text)))
    n = len(out)
    summary = dict(n=n, acc=sum(o['acc'] for o in out) / n, fmt=sum(o['fmt'] for o in out) / n,
                   length=sum(o['length'] for o in out) / n)
    return summary, out
