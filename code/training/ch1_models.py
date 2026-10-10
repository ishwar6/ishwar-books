"""Shared loader for the Chapter 1 experiments: Qwen2.5-0.5B (base) and Qwen2.5-0.5B-Instruct, in float32 on the Apple GPU."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = 'Qwen/Qwen2.5-0.5B'
INSTRUCT = 'Qwen/Qwen2.5-0.5B-Instruct'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


def load(name, dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV).eval()
    return tok, model


@torch.no_grad()
def greedy(tok, model, ids, max_new_tokens=120):
    """Greedy decoding (always take the most likely next token). Returns the new token ids and whether it stopped by itself."""
    stop_ids = {tok.convert_tokens_to_ids(t) for t in ['<|endoftext|>', '<|im_end|>']}   # the same stop tokens for both models
    ids = torch.tensor([ids], device=DEV)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
                         repetition_penalty=1.0,   # the instruct model ships with 1.1; pure greedy means none
                         eos_token_id=sorted(stop_ids), pad_token_id=tok.convert_tokens_to_ids('<|endoftext|>'))
    new = out[0, ids.shape[1]:].tolist()
    stopped = any(t in stop_ids for t in new)
    if stopped:
        new = new[:next(i for i, t in enumerate(new) if t in stop_ids) + 1]
    return new, stopped
