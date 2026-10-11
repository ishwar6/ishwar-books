"""Shared helpers for Chapter 7 (reward models).

Data: UltraFeedback binarized (HuggingFaceH4/ultrafeedback_binarized). Each row is a prompt, a "chosen" response (the one
GPT-4 rated highest out of four) and a "rejected" response (one of the other three, picked at random), with both ratings
(1 to 10). We keep single-turn pairs with different ratings whose full chats fit in MAXLEN tokens.

Model: a reward model = the Qwen2.5-0.5B-Instruct transformer + a linear "score" head that maps the hidden state of the
last token to one number. The transformer is adapted with LoRA; the head is trained in full."""
import os, random
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BACKBONE = 'Qwen/Qwen2.5-0.5B-Instruct'
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
MAXLEN = 512
HERE = os.path.dirname(os.path.abspath(__file__))
RM_DIR = os.path.join(HERE, 'results', 'ch7_rm')          # the saved adapter that Chapter 8 loads
CACHE = os.path.expanduser('~/.cache/ch7_reward_models')  # extra adapters (ablations), not committed
os.makedirs(CACHE, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BACKBONE)
PAD = tok.pad_token_id                                    # <|endoftext|>, never appears inside our chats


def parquet(split):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download('HuggingFaceH4/ultrafeedback_binarized', f'data/{split}-00000-of-00001.parquet', repo_type='dataset')
    return pd.read_parquet(p)


def chat_text(prompt, response):
    """The full chat as one string, in Qwen's template. We drop the final newline so the last token is <|im_end|>:
    the reward is read from the hidden state at the token that closes the assistant's turn."""
    msgs = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    return tok.apply_chat_template(msgs, tokenize=False).rstrip('\n')


def load_pairs(split, n, seed=0, maxlen=MAXLEN):
    """n preference pairs from 'train_prefs' or 'test_prefs': ties dropped, single turn only, both chats <= maxlen tokens."""
    d = parquet(split)
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        r = d.iloc[i]
        if r.score_chosen == r.score_rejected or len(r.chosen) != 2 or len(r.rejected) != 2:
            continue
        c, j = r.chosen[1]['content'], r.rejected[1]['content']
        if not c.strip() or not j.strip() or c == j:
            continue
        tc, tj = chat_text(r.prompt, c), chat_text(r.prompt, j)
        ic, ij = tok(tc)['input_ids'], tok(tj)['input_ids']
        if len(ic) > maxlen or len(ij) > maxlen:
            continue
        out.append(dict(prompt=r.prompt, chosen=c, rejected=j, sc=float(r.score_chosen), sr=float(r.score_rejected),
                        ids_c=ic, ids_r=ij, words_c=len(c.split()), words_r=len(j.split()),
                        ntok_c=len(tok(c)['input_ids']), ntok_r=len(tok(j)['input_ids'])))
        if len(out) == n:
            break
    return out


def pad(seqs, bucket=64):
    """Right-pad token lists to one length, rounded up to a multiple of `bucket` (fewer tensor shapes = fewer
    kernel compilations on the Apple GPU)."""
    L = max(len(s) for s in seqs)
    L = -(-L // bucket) * bucket
    ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        att[i, :len(s)] = 1
    return ids.to(DEV), att.to(DEV)


def build_rm(r=16, alpha=32, dropout=0.05, seed=0):
    """Backbone + scalar head, with LoRA adapters on every linear layer of the transformer."""
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    m = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=1, dtype=torch.float32)
    m.config.pad_token_id = PAD                       # tells the model where each sequence ends
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, task_type='SEQ_CLS',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                     modules_to_save=['score'])
    return get_peft_model(m, cfg).to(DEV)


def load_rm(path=RM_DIR):
    """Load a saved reward model adapter on top of the backbone (this is what Chapter 8 does)."""
    from peft import PeftModel
    m = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=1, dtype=torch.float32)
    m.config.pad_token_id = PAD
    return PeftModel.from_pretrained(m, path).to(DEV).eval()


@torch.no_grad()
def score_ids(model, seqs, bs=16):
    """Rewards for a list of token-id lists, shortest first for speed, returned in the original order."""
    model.eval()
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    for k in range(0, len(order), bs):
        chunk = order[k:k + bs]
        ids, att = pad([seqs[i] for i in chunk])
        r = model(input_ids=ids, attention_mask=att).logits[:, 0].float().cpu()
        for i, v in zip(chunk, r.tolist()):
            out[i] = v
    return out


def score_texts(model, prompts, responses, bs=16):
    seqs = [tok(chat_text(p, y))['input_ids'][-MAXLEN:] for p, y in zip(prompts, responses)]
    return score_ids(model, seqs, bs)


GAP_EDGES = (1, 2, 4)                                     # rating-gap buckets: [0,1) [1,2) [2,4) [4,10)
GAP_NAMES = ('negligibly better', 'slightly better', 'better', 'significantly better')


def gap_bucket(gap):
    return sum(gap >= e for e in GAP_EDGES)


def margin_of(gap, scale=1.0):
    """Llama 2's "margin small" (Table 27): 0, 1/3, 2/3, 1 for negligibly / slightly / plain / significantly better.
    scale=3 gives their "margin large" (0, 1, 2, 3). We map UltraFeedback rating gaps onto the four buckets."""
    return scale * gap_bucket(gap) / 3


def bt_loss(rc, rr, margin=None):
    """The Bradley-Terry reward model loss, -log sigmoid(r_chosen - r_rejected [- margin]), averaged over the batch."""
    d = rc - rr if margin is None else rc - rr - margin
    return -F.logsigmoid(d).mean()


# ------------------------------------------------------------------ the "gold" reward model
GOLD = 'Skywork/Skywork-Reward-V2-Qwen3-0.6B'      # a 2025 reward model trained on far more preference data than ours


def load_gold():
    gt = AutoTokenizer.from_pretrained(GOLD)
    gm = AutoModelForSequenceClassification.from_pretrained(GOLD, num_labels=1, dtype=torch.float32).to(DEV).eval()
    gm.config.pad_token_id = gt.pad_token_id
    return gt, gm


@torch.no_grad()
def score_gold(gt, gm, prompts, responses, bs=8, maxlen=1024):
    seqs = []
    for p, y in zip(prompts, responses):
        conv = [{'role': 'user', 'content': p}, {'role': 'assistant', 'content': y}]
        ids = gt.apply_chat_template(conv, tokenize=True)
        ids = ids['input_ids'] if isinstance(ids, dict) or hasattr(ids, 'keys') else ids
        seqs.append(list(ids)[-maxlen:])
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = [0.0] * len(seqs)
    for k in range(0, len(order), bs):
        chunk = order[k:k + bs]
        L = -(-max(len(seqs[i]) for i in chunk) // 64) * 64
        ids = torch.full((len(chunk), L), gt.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, i in enumerate(chunk):
            ids[r, :len(seqs[i])] = torch.tensor(seqs[i]); att[r, :len(seqs[i])] = 1
        v = gm(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).logits[:, 0].float().cpu().tolist()
        for i, x in zip(chunk, v):
            out[i] = x
    return out


def load_many(names):
    """One backbone with several reward model adapters on it; switch with m.set_adapter(name)."""
    from peft import PeftModel
    base = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=1, dtype=torch.float32)
    base.config.pad_token_id = PAD
    path = lambda n: RM_DIR if n == 'main' else os.path.join(CACHE, n)
    m = PeftModel.from_pretrained(base, path(names[0]), adapter_name=names[0])
    for n in names[1:]:
        m.load_adapter(path(n), adapter_name=n)
    return m.to(DEV).eval()
