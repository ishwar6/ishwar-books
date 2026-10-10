"""Shared helpers for the Chapter 3 experiments (how we measure a model).
Models: Qwen2.5-0.5B (base) and Qwen2.5-0.5B-Instruct, float32 on the Apple GPU.
Datasets: downloaded once from the Hugging Face Hub as parquet files (MMLU, GSM8K, HumanEval, RewardBench, WikiText-2)."""
import random
import torch
import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = 'Qwen/Qwen2.5-0.5B'
INSTRUCT = 'Qwen/Qwen2.5-0.5B-Instruct'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'

FILES = {
    'wikitext2_test': ('Salesforce/wikitext', 'wikitext-2-raw-v1/test-00000-of-00001.parquet'),
    'mmlu_test': ('cais/mmlu', 'all/test-00000-of-00001.parquet'),
    'mmlu_dev': ('cais/mmlu', 'all/dev-00000-of-00001.parquet'),
    'gsm8k_test': ('openai/gsm8k', 'main/test-00000-of-00001.parquet'),
    'gsm8k_train': ('openai/gsm8k', 'main/train-00000-of-00001.parquet'),
    'humaneval': ('openai/openai_humaneval', 'openai_humaneval/test-00000-of-00001.parquet'),
    'rewardbench': ('allenai/reward-bench', 'data/filtered-00000-of-00001.parquet'),
}


def data(name):
    repo, path = FILES[name]
    return pd.read_parquet(hf_hub_download(repo, path, repo_type='dataset'))


def load(name, dtype=torch.float32, side='left'):
    tok = AutoTokenizer.from_pretrained(name, padding_side=side)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV).eval()
    return tok, model


def chat(tok, user, system=None):
    """The prompt text for one user turn in the model's chat template, ending where the assistant starts writing."""
    msgs = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def generate(tok, model, prompts, max_new_tokens=256, batch=16, do_sample=False, temperature=1.0, top_p=1.0, seed=0):
    """Batched generation with left padding. Returns the decoded new text for each prompt (cut at the first stop token)."""
    stop = [tok.convert_tokens_to_ids(t) for t in ['<|endoftext|>', '<|im_end|>']]
    pad = tok.convert_tokens_to_ids('<|endoftext|>')
    outs = []
    torch.manual_seed(seed)
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors='pt', padding=True).to(DEV)
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=None) if do_sample else \
            dict(do_sample=False, temperature=None, top_p=None, top_k=None)
        g = model.generate(**enc, max_new_tokens=max_new_tokens, eos_token_id=stop, pad_token_id=pad, **kw)
        for row in g[:, enc['input_ids'].shape[1]:].tolist():
            cut = next((j for j, t in enumerate(row) if t in stop), len(row))
            outs.append(tok.decode(row[:cut]))
    return outs


def sample_rows(df, n, seed=0):
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    return df.iloc[sorted(idx[:n])].reset_index(drop=True)
