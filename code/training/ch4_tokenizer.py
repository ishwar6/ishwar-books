"""Chapter 4: byte-pair encoding from scratch on the toy vocabulary of Sennrich et al. (2016), then three real tokenizers
(GPT-2, Qwen2.5, and the 4,096-token BPE trained for the TinyStories run) on the same strings."""
import collections, re
from transformers import AutoTokenizer
from common import Log, save
from ch4_gpt import get_tokenizer

log = Log('ch4_tokenizer')
R = {}
# ---- 1. BPE from scratch (Sennrich et al. 2016, Algorithm 1). Words are split into characters, </w> marks a word end.
vocab = {'l o w </w>': 5, 'l o w e r </w>': 2, 'n e w e s t </w>': 6, 'w i d e s t </w>': 3}


def pair_counts(vocab):
    pairs = collections.Counter()
    for word, freq in vocab.items():
        syms = word.split()
        for a, b in zip(syms, syms[1:]):
            pairs[a, b] += freq                    # count each adjacent pair, weighted by word frequency
    return pairs


def merge(pair, vocab):
    pat = re.compile(r'(?<!\S)' + re.escape(' '.join(pair)) + r'(?!\S)')
    return {pat.sub(''.join(pair), w): f for w, f in vocab.items()}   # glue the pair into one symbol everywhere


log('== BPE from scratch ==')
log('start:', vocab)
R['bpe'] = []
for i in range(10):
    pairs = pair_counts(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge(best, vocab)
    log(f'merge {i + 1:2d}: {best[0]!r} + {best[1]!r}  (seen {pairs[best]} times)  ->  {list(vocab)}')
    R['bpe'].append({'pair': best, 'count': pairs[best], 'vocab': list(vocab)})

# ---- 2. Real tokenizers on the same strings
log('')
log('== three real tokenizers ==')
toks = {'GPT-2 (50,257)': AutoTokenizer.from_pretrained('gpt2'),
        'Qwen2.5 (151,665)': AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B')}
ts = get_tokenizer(4096)
strings = ['The cat sat on the mat.', 'Pretraining is unbelievably expensive.', 'def add(a, b): return a + b',
           'The year 2024 had 366 days.', 'नमस्ते, आप कैसे हैं?', 'Once upon a time, a little girl named Lily']
R['real'] = []
for s in strings:
    log(f'text: {s!r}  ({len(s)} characters)')
    row = {'text': s}
    for name, t in toks.items():
        ids = t.encode(s)
        pieces = [t.decode([i]) for i in ids]
        log(f'   {name:18s} {len(ids):3d} tokens: ' + ' | '.join(pieces))
        row[name] = pieces
    enc = ts.encode(s)
    pieces = [ts.decode([i]) for i in enc.ids]
    log(f'   {"TinyStories (4,096)":18s} {len(enc.ids):3d} tokens: ' + ' | '.join(pieces))
    row['TinyStories (4,096)'] = pieces
    R['real'].append(row)
q = toks['Qwen2.5 (151,665)']
log('')
log(f'Qwen2.5 vocabulary size: {len(q):,}; ids of " the", " The", "the": {q.encode(" the")}, {q.encode(" The")}, {q.encode("the")}')
save('ch4_tokenizer', R)
