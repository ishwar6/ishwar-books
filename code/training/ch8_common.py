"""Shared helpers for Chapter 8 (RLHF with PPO).

The four models of RLHF, on one Apple GPU:
  policy     Qwen2.5-0.5B-Instruct + a LoRA adapter (the only part of it that trains)
  reference  the same backbone with the adapter switched off (so it costs no extra memory)
  reward     the Chapter 7 reward model (backbone + its LoRA adapter + score head), frozen
  value      a second copy of the Chapter 7 reward model whose adapter and head DO train, read at every token
"""
import json, os, random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from ch7_common import tok, BACKBONE, DEV, PAD, RM_DIR, CACHE, parquet, load_pairs, chat_text, MAXLEN

HERE = os.path.dirname(os.path.abspath(__file__))
STOP = [tok.convert_tokens_to_ids(t) for t in ['<|im_end|>', '<|endoftext|>']]
IM_END = STOP[0]
LORA_TARGETS = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
PROMPTS = os.path.join(CACHE, 'ch8_prompts.json')        # written by ch8_prompts.py
CKPT = os.path.expanduser('~/.cache/ch8_ppo')              # policy adapters saved during training (not committed)
os.makedirs(CKPT, exist_ok=True)


def prompt_ids(p):
    """The policy's input: the user turn in Qwen's chat template (default system line), ending with the assistant header."""
    text = tok.apply_chat_template([{'role': 'user', 'content': p}], tokenize=False, add_generation_prompt=True)
    return tok(text)['input_ids']


def load_policy(adapter=None, r=16, alpha=32, seed=0):
    """Qwen2.5-0.5B-Instruct with a LoRA adapter: fresh (B = 0, so it starts exactly at the reference) or loaded."""
    from peft import LoraConfig, get_peft_model, PeftModel
    torch.manual_seed(seed)
    base = AutoModelForCausalLM.from_pretrained(BACKBONE, dtype=torch.float32)
    if adapter:
        m = PeftModel.from_pretrained(base, adapter, is_trainable=False)
    else:
        m = get_peft_model(base, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules=LORA_TARGETS,
                                            task_type='CAUSAL_LM'))
    return m.to(DEV)


@torch.no_grad()
def generate(model, prompts, max_new, temp, n=1):
    """Sample n replies per prompt. Returns a list of dicts: prompt ids, reply ids (without the stop token),
    finished (did it emit <|im_end|> within max_new tokens?)."""
    model.eval()
    tok.padding_side = 'left'
    items = [(p, prompt_ids(p)) for p in prompts for _ in range(n)]
    L = max(len(i) for _, i in items)
    ids = torch.full((len(items), L), PAD, dtype=torch.long)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for k, (_, i) in enumerate(items):
        ids[k, L - len(i):] = torch.tensor(i)
        att[k, L - len(i):] = 1
    g = model.generate(input_ids=ids.to(DEV), attention_mask=att.to(DEV), max_new_tokens=max_new, do_sample=True,
                       temperature=temp, top_p=1.0, top_k=0, eos_token_id=STOP, pad_token_id=PAD)
    out = []
    for (p, pi), row in zip(items, g[:, L:].tolist()):
        cut = next((j for j, t in enumerate(row) if t in STOP), None)
        out.append(dict(prompt=p, p_ids=pi, r_ids=row[:cut] if cut is not None else row, finished=cut is not None))
    return out


def batch_tensors(samples):
    """Right-padded tensors for [prompt + reply (+ <|im_end|> if it finished)], plus a mask of the reply positions
    whose tokens the policy chose (the <|im_end|> that closes a finished reply is an action too)."""
    seqs, masks = [], []
    for s in samples:
        r = s['r_ids'] + ([IM_END] if s['finished'] else [])
        seqs.append(s['p_ids'] + r)
        masks.append([0] * len(s['p_ids']) + [1] * len(r))
    L = max(len(x) for x in seqs)
    L = -(-L // 64) * 64                    # round up to a multiple of 64: fewer tensor shapes, fewer kernel compilations
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    act = torch.zeros((len(seqs), L), dtype=torch.float32)
    for k, (x, m) in enumerate(zip(seqs, masks)):
        ids[k, :len(x)] = torch.tensor(x)
        att[k, :len(x)] = 1
        act[k, :len(m)] = torch.tensor(m, dtype=torch.float32)
    # shift by one: position t of the outputs predicts token t+1, so action masks move one place left
    return ids.to(DEV), att.to(DEV), act[:, 1:].to(DEV)


def token_logprobs(model, ids, att, temp, with_entropy=False):
    """log pi(token_t+1 | tokens <= t) for every position, from logits divided by the sampling temperature
    (the distribution the tokens were actually drawn from)."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float() / temp
    logp_all = F.log_softmax(logits, -1)
    lp = logp_all.gather(-1, ids[:, 1:, None])[..., 0]
    if with_entropy:
        ent = -(logp_all.exp() * logp_all).sum(-1)
        return lp, ent
    return lp


def rm_texts(samples):
    """What the reward model reads: the chat in its training format, ending with <|im_end|> (also for cut-off replies)."""
    return [tok(chat_text(s['prompt'], tok.decode(s['r_ids'])))['input_ids'][-MAXLEN:] for s in samples]


def load_prompts():
    return json.load(open(PROMPTS))
