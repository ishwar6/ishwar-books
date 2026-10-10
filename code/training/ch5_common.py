"""Shared helpers for Chapter 5 (supervised fine-tuning): the Alpaca-cleaned data in Qwen's chat template with the prompt
masked out of the loss, a padding collator, the held-out loss, the instruction-following checks, and two forgetting probes."""
import json, math, random, re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = 'Qwen/Qwen2.5-0.5B'
INSTRUCT = 'Qwen/Qwen2.5-0.5B-Instruct'
DEV = 'mps'
SYSTEM = 'You are a helpful assistant.'
tok = AutoTokenizer.from_pretrained(BASE)
END = tok.convert_tokens_to_ids('<|im_end|>')          # 151645, closes every turn
EOT = tok.convert_tokens_to_ids('<|endoftext|>')       # 151643, the base model's end of document


def messages(ex):
    """One Alpaca row -> a chat: the instruction (plus the optional input) is the user turn, the output is the reply."""
    user = ex['instruction'] + ('\n\n' + ex['input'] if ex['input'].strip() else '')
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}], ex['output']


def encode(ex, max_len=512):
    """Token ids and labels. The prompt part gets label -100 (ignored by the loss); the reply and its <|im_end|> are learned."""
    msgs, reply = messages(ex)
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)   # ends with "<|im_start|>assistant\n"
    prompt = tok(text)['input_ids']
    answer = tok(reply)['input_ids'] + [END]
    ids = (prompt + answer)[:max_len]
    labels = ([-100] * len(prompt) + answer)[:max_len]
    return {'input_ids': ids, 'labels': labels}


def load_split(n_train=1000, n_val=100, seed=0, max_len=384):
    from datasets import load_dataset
    d = load_dataset('yahma/alpaca-cleaned', split='train')
    idx = list(range(len(d)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx:
        e = encode(d[i], max_len)
        if len(e['input_ids']) < max_len and sum(l != -100 for l in e['labels']) >= 2:   # keep only examples that fit whole
            out.append((d[i], e))
        if len(out) == n_train + n_val:
            break
    return out[:n_train], out[n_train:]


def collate(batch, bucket=128):
    """Pad a list of encoded examples to the longest one, rounded up to a multiple of `bucket` tokens.
    Padding is masked twice: attention_mask 0 and label -100. The rounding keeps the number of distinct tensor shapes small:
    the Apple GPU backend prepares (compiles) its kernels for each new shape, and with a new shape at every step training
    ran about ten times slower."""
    L = max(len(e['input_ids']) for e in batch)
    L = -(-L // bucket) * bucket
    ids = torch.full((len(batch), L), EOT, dtype=torch.long)
    lab = torch.full((len(batch), L), -100, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    for i, e in enumerate(batch):
        n = len(e['input_ids'])
        ids[i, :n] = torch.tensor(e['input_ids'])
        lab[i, :n] = torch.tensor(e['labels'])
        att[i, :n] = 1
    return ids, lab, att


def sft_loss(model, ids, lab, att, reduction='mean'):
    """Cross-entropy over the reply tokens only. The logits at position t predict token t+1, so the labels are shifted by one;
    positions labelled -100 (prompt and padding) are ignored. Every tensor keeps a fixed shape (no boolean indexing), because the
    Apple GPU backend prepares new kernels for every new shape."""
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1]
    target = lab[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), target.reshape(-1), ignore_index=-100, reduction=reduction)


@torch.no_grad()
def heldout_loss(model, val, bs=8):
    """Mean loss per reply token over the held-out examples."""
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(val), bs):
        ids, lab, att = (t.to(DEV) for t in collate([e for _, e in val[i:i + bs]]))
        with torch.autocast('mps', dtype=torch.bfloat16):
            tot += sft_loss(model, ids, lab, att, reduction='sum').item()
        n += (lab[:, 1:] != -100).sum().item()
    return tot / n


@torch.no_grad()
def chat(model, user, max_new=150):
    """Greedy reply in the chat template. Returns (text, stopped): stopped is True if the model ended its turn itself."""
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    p = tok(text, return_tensors='pt').input_ids.to(DEV)
    out = model.generate(input_ids=p, attention_mask=torch.ones_like(p), max_new_tokens=max_new, do_sample=False,
                         eos_token_id=[END, EOT], pad_token_id=EOT)
    new = out[0, p.shape[1]:].tolist()
    stopped = len(new) > 0 and new[-1] in (END, EOT)
    return tok.decode(new, skip_special_tokens=True).strip(), stopped


def _sentences(t):
    return [s for s in re.split(r'(?<=[.!?])\s+', t.strip()) if s.strip()]


def _json_ok(t):
    m = re.search(r'\{.*\}', t, re.S)
    try:
        o = json.loads(m.group(0)) if m else None
        return isinstance(o, dict) and 'name' in o and 'age' in o
    except Exception:
        return False


CHECKS = [
    ('List three fruits as a bulleted list with exactly three items and nothing else.',
     lambda t: len([l for l in t.splitlines() if re.match(r'\s*([-*•]|\d+[.)])\s+', l)]) == 3),
    ('Answer in one word: what colour is the sky on a clear day?', lambda t: len(re.findall(r'\w+', t)) <= 2),
    ("Write the word 'hello' in capital letters and nothing else.", lambda t: t.strip().strip('.!"\'') == 'HELLO'),
    ('Describe a cat in one sentence, using only lowercase letters.', lambda t: t == t.lower() and len(_sentences(t)) == 1),
    ('What is 12 + 30? Reply with just the number.', lambda t: re.fullmatch(r'\s*42\.?\s*', t) is not None),
    ("Translate 'good morning' into French. Give only the translation.", lambda t: 'bonjour' in t.lower() and len(t.split()) <= 4),
    ('Write a haiku about rain. Use exactly three lines.', lambda t: len([l for l in t.splitlines() if l.strip()]) == 3),
    ('Give a JSON object with the keys "name" and "age" for a person called Ana who is 30.', _json_ok),
    ('In one sentence, what is the capital of Japan?', lambda t: 'tokyo' in t.lower() and len(_sentences(t)) == 1),
    ('Write exactly two sentences about the moon.', lambda t: len(_sentences(t)) == 2),
    ('Explain photosynthesis in fewer than 30 words.', lambda t: 0 < len(t.split()) < 30),
    ("Start your answer with the word 'Yes'. Is water wet?", lambda t: t.strip().lower().startswith('yes')),
]


def run_checks(model, log=None):
    rows = []
    for q, ok in CHECKS:
        t, stopped = chat(model, q)
        passed = bool(stopped and ok(t))
        rows.append({'prompt': q, 'reply': t, 'stopped': stopped, 'passed': passed})
        if log:
            log(f'  [{"PASS" if passed else "fail"}] {"stops " if stopped else "RUNS ON"} {q[:60]!r} -> {t[:110]!r}')
    return rows


@torch.no_grad()
def wikitext_ppl(model, n_windows=40, T=512):
    """Perplexity on plain Wikipedia text (wikitext-2 test), a probe of general language modelling, with no chat template."""
    from datasets import load_dataset
    text = '\n'.join(load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')['text'])
    ids = tok(text)['input_ids'][:n_windows * T]
    model.eval()
    tot = 0.0
    for i in range(n_windows):
        x = torch.tensor([ids[i * T:(i + 1) * T]], device=DEV)
        with torch.autocast('mps', dtype=torch.bfloat16):
            logits = model(input_ids=x).logits[:, :-1].float()
        tot += F.cross_entropy(logits[0], x[0, 1:]).item()
    return math.exp(tot / n_windows)


ANT = [('hot', 'cold'), ('big', 'small'), ('fast', 'slow'), ('happy', 'sad'), ('up', 'down'), ('early', 'late'), ('rich', 'poor'),
       ('strong', 'weak'), ('full', 'empty'), ('wet', 'dry'), ('good', 'bad'), ('loud', 'quiet'), ('first', 'last'), ('true', 'false'),
       ('win', 'lose'), ('push', 'pull'), ('give', 'take'), ('buy', 'sell'), ('love', 'hate'), ('clean', 'dirty'), ('thick', 'thin'),
       ('high', 'low'), ('cheap', 'expensive'), ('inside', 'outside'), ('always', 'never'), ('remember', 'forget'), ('arrive', 'leave'),
       ('safe', 'dangerous'), ('wide', 'narrow'), ('open', 'closed')]


@torch.no_grad()
def fewshot_antonyms(model, k=4):
    """The 4-shot antonym task of Chapter 4, as a plain-text prompt (no chat template): does few-shot ability survive SFT?"""
    pool, test = ANT[:k], ANT[k:]
    head = 'Write the opposite of each word:\n' + ''.join(f'{x} -> {y}\n' for x, y in pool)
    right = 0
    for x, y in test:
        p = tok(head + f'{x} ->', return_tensors='pt').input_ids.to(DEV)
        out = model.generate(input_ids=p, attention_mask=torch.ones_like(p), max_new_tokens=4, do_sample=False, pad_token_id=EOT)
        right += tok.decode(out[0, p.shape[1]:]).strip().lower().startswith(y)
    return right / len(test)


def load(name, dtype=torch.float32):
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(DEV)
