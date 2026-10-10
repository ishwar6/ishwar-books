"""Chapter 3, experiment 1: perplexity and bits per byte on held-out text.
A. WikiText-2 test set (Wikipedia articles), scored with a sliding window, for GPT-2 (124M), Qwen2.5-0.5B base and Instruct.
   Perplexity depends on the tokenizer; bits per byte does not, so only BPB can compare GPT-2 with Qwen.
B. The same model, three window settings: the number changes with how you measure it.
C. "Perplexity is not usefulness": both Qwen models on assistant-style text (GSM8K reference solutions in the chat
   template), with the loss counted only on the answer tokens."""
import math, re, time
import torch
import torch.nn.functional as Fn
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import Log, save
from ch3_common import BASE, INSTRUCT, DEV, data, load, chat, sample_rows

log = Log('ch3_perplexity')
LN2 = math.log(2)


@torch.no_grad()
def sliding_nll(model, ids, window=1024, stride=512):
    """Sum of -log p over every token after the first, scoring each token once.
    Each window sees up to `window` tokens; only the last `stride` of them (the new ones) are scored,
    so every scored token has at least window - stride tokens of context (except at the very start)."""
    total, count, prev_end = 0.0, 0, 0
    for begin in range(0, len(ids), stride):
        end = min(begin + window, len(ids))
        x = torch.tensor([ids[begin:end]], device=DEV)
        logits = model(x).logits[0, :-1].float()                 # position t predicts token t+1
        nll = Fn.cross_entropy(logits, x[0, 1:], reduction='none')
        new = end - prev_end                                     # tokens not scored by an earlier window
        nll = nll[-new:] if begin > 0 else nll                   # the first token of the text has no prediction
        total += nll.sum().item(); count += nll.numel()
        prev_end = end
        if end == len(ids):
            break
    return total, count


text = '\n\n'.join(data('wikitext2_test')['text'])               # the usual way: join all lines of the test split
nbytes = len(text.encode('utf-8'))
log('== A. WikiText-2 test set ==')
log(f'characters: {len(text):,}   UTF-8 bytes: {nbytes:,}')
res = {'bytes': nbytes, 'chars': len(text), 'A': {}, 'B': [], 'C': {}}
models = [('gpt2', 'GPT-2 (124M)'), (BASE, 'Qwen2.5-0.5B base'), (INSTRUCT, 'Qwen2.5-0.5B-Instruct')]
for name, label in models:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(DEV).eval()
    ids = tok(text)['input_ids']
    t0 = time.time()
    nll, n = sliding_nll(model, ids, 1024, 512)
    loss = nll / n
    r = dict(tokens=len(ids), scored=n, nll_sum=nll, loss=loss, ppl=math.exp(loss),
             bits_per_token=loss / LN2, bytes_per_token=nbytes / len(ids), bpb=nll / LN2 / nbytes, secs=time.time() - t0)
    res['A'][label] = r
    log(f'{label:24s} tokens {len(ids):>7,}  bytes/token {r["bytes_per_token"]:.2f}  loss {loss:.4f} nats/token  '
        f'PPL {r["ppl"]:7.2f}  bits/byte {r["bpb"]:.4f}  ({r["secs"]:.0f}s)')
    if name == BASE:
        log('')
        log('== B. same model, different measuring choices (Qwen2.5-0.5B base) ==')
        for window, stride in [(1024, 1024), (1024, 512), (256, 256), (2048, 1024)]:
            nll2, n2 = sliding_nll(model, ids, window, stride)
            r2 = dict(window=window, stride=stride, loss=nll2 / n2, ppl=math.exp(nll2 / n2), bpb=nll2 / LN2 / nbytes)
            res['B'].append(r2)
            log(f'window {window:5d} stride {stride:5d}:  loss {r2["loss"]:.4f}  PPL {r2["ppl"]:6.2f}  bits/byte {r2["bpb"]:.4f}')
        log('')
    del model
    if DEV == 'mps':
        torch.mps.empty_cache()

# ---------------------------------------------------------------- C. assistant-style text
log('== C. assistant-style text: 200 GSM8K test solutions in the chat template, loss on answer tokens only ==')
rows = sample_rows(data('gsm8k_test'), 200, seed=1)
for name, label in [(BASE, 'Qwen2.5-0.5B base'), (INSTRUCT, 'Qwen2.5-0.5B-Instruct')]:
    tok, model = load(name)
    tot, cnt = 0.0, 0
    for q, a in zip(rows['question'], rows['answer']):
        p = tok(chat(tok, q))['input_ids']
        full = p + tok(re.sub(r'<<[^>]*>>', '', a) + '<|im_end|>')['input_ids']
        x = torch.tensor([full], device=DEV)
        with torch.no_grad():
            logits = model(x).logits[0, :-1].float()
        nll = Fn.cross_entropy(logits, x[0, 1:], reduction='none')[len(p) - 1:]   # only the answer tokens
        tot += nll.sum().item(); cnt += nll.numel()
    res['C'][label] = dict(loss=tot / cnt, ppl=math.exp(tot / cnt), tokens=cnt)
    log(f'{label:24s} answer tokens {cnt:,}  loss {tot / cnt:.4f}  PPL {math.exp(tot / cnt):.2f}')
    del model
save('ch3_perplexity', res)
