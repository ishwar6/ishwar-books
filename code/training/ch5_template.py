"""Chapter 5: the SFT loss with label masking, worked on real tokens. (1) A two-turn chat in Qwen's template and its labels.
(2) Per-token losses of the base model on one Alpaca example: the masked loss (reply only) against the unmasked one, and the
cost of the <|im_end|> token. (3) Padding versus packing for our 2,000 training examples. Runs on the CPU."""
import os, torch, torch.nn.functional as F, numpy as np
from transformers import AutoModelForCausalLM
from common import Log, save
from ch5_common import tok, END, SYSTEM, encode, load_split

log = Log('ch5_template')
R = {}
# ---- 1. a two-turn conversation: which tokens are learned?
conv = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'Name a primary colour.'},
        {'role': 'assistant', 'content': 'Red.'}, {'role': 'user', 'content': 'Another one?'},
        {'role': 'assistant', 'content': 'Blue.'}]
text = tok.apply_chat_template(conv, tokenize=False)
ids = tok(text)['input_ids']
# label a token if it lies inside an assistant reply (its content and the closing <|im_end|>)
labels, in_reply = [], False
pieces = [tok.decode([i]) for i in ids]
for k, (i, p) in enumerate(zip(ids, pieces)):
    if k >= 2 and pieces[k - 2] == '<|im_start|>' and pieces[k - 1] == 'assistant':
        in_reply = True                            # this is the "\n" after "assistant": the reply starts on the next token
        labels.append(-100); continue
    labels.append(i if in_reply else -100)
    if in_reply and i == END:
        in_reply = False
log('== two-turn chat in the Qwen2.5 template ==')
log(text.replace('\n', '\\n\n'))
log('token-by-token labels (L = learned, . = masked):')
log(' '.join(('L:' if l != -100 else '.:') + repr(p) for p, l in zip(pieces, labels)))
log(f'{len(ids)} tokens, {sum(l != -100 for l in labels)} learned')
R['multi'] = {'pieces': pieces, 'learned': [l != -100 for l in labels]}

# ---- 2. per-token losses of the base model on one example
m = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.float32).eval()
ex = {'instruction': 'Name three primary colours.', 'input': '', 'output': 'The three primary colours are red, blue and yellow.'}
e = encode(ex)
x = torch.tensor([e['input_ids']])
lab = torch.tensor([e['labels']])


def per_token(model):
    with torch.no_grad():
        logits = model(x).logits[0, :-1]
    return F.cross_entropy(logits, x[0, 1:], reduction='none')


def report(name, lt):
    mask = lab[0, 1:] != -100
    log(f'-- {name}')
    rows = []
    for t in range(len(lt)):
        if mask[t]:
            p = tok.decode([x[0, t + 1].item()])
            rows.append([p, lt[t].item()])
            log(f'   predict {p!r:14s} loss {lt[t].item():7.3f}   p = {torch.exp(-lt[t]).item():.4f}')
    masked = lt[mask].mean().item()
    full = lt.mean().item()
    log(f'   masked loss (reply tokens only, {mask.sum().item()} tokens) = {masked:.3f}')
    log(f'   unmasked loss (all {len(lt)} predicted tokens)            = {full:.3f}')
    log(f'   masked loss without the final <|im_end|>                 = {lt[mask][:-1].mean().item():.3f}')
    return {'rows': rows, 'masked': masked, 'full': full, 'no_end': lt[mask][:-1].mean().item()}


log('')
log('== per-token loss on one example ==')
log(repr(tok.decode(e['input_ids'])))
R['base'] = report('Qwen2.5-0.5B base', per_token(m))
from peft import PeftModel
for key, name in [('lora_plain', 'after plain LoRA SFT (ch5_sft.py lora_plain)'), ('lora', 'after LoRA SFT with the two token rows (ch5_sft.py lora)')]:
    adapter = f'/Users/admin/.cache/ch5_{key}_adapter'
    if os.path.exists(adapter):
        base = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.float32).eval()
        pm = PeftModel.from_pretrained(base, adapter).eval()
        R[key] = report(name, per_token(pm))

# ---- 3. padding versus packing
train, _ = load_split()
L = np.array([len(e['input_ids']) for _, e in train])
rng = np.random.default_rng(1)
order = rng.permutation(len(L))
pad_tokens = sum(L[order[i:i + 8]].max() * len(order[i:i + 8]) for i in range(0, len(L), 8))
bucket_tokens = sum(-(-L[order[i:i + 8]].max() // 128) * 128 * len(order[i:i + 8]) for i in range(0, len(L), 8))
real = L.sum()
reply = sum(sum(l != -100 for l in e['labels']) for _, e in train)
packed_rows = int(np.ceil(real / 384))
log('')
log(f'== padding versus packing for the {len(train):,} training examples ==')
log(f'lengths: mean {L.mean():.1f}, median {np.median(L):.0f}, max {L.max()} tokens; total {real:,} real tokens')
log(f'reply tokens (the ones the loss is computed on): {reply:,}')
log(f'random batches of 8, padded to the longest: {pad_tokens:,} token slots, {1 - real / pad_tokens:.1%} of them padding')
log(f'same, with the length rounded up to a multiple of 128 (as in ch5_sft.py): {bucket_tokens:,} slots, {1 - real / bucket_tokens:.1%} padding')
log(f'packed into rows of 384: {packed_rows} rows = {packed_rows * 384:,} slots, {1 - real / (packed_rows * 384):.1%} padding')
R['pack'] = {'mean': float(L.mean()), 'median': float(np.median(L)), 'max': int(L.max()), 'real': int(real),
             'pad_slots': int(pad_tokens), 'packed_slots': packed_rows * 384, 'bucket_slots': int(bucket_tokens), 'reply': int(reply), 'hist': np.histogram(L, bins=range(0, 541, 30))[0].tolist()}
save('ch5_template', R)
