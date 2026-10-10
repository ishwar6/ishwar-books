"""Chapter 4: a small FineWeb-style pipeline on real Common Crawl text (the first 12 MB of one WET file of CC-MAIN-2024-33).
Steps: language filter -> Gopher quality + repetition rules -> C4 rules (without terminal punctuation) -> FineWeb custom rules
-> MinHash near-duplicate detection -> the FineWeb-Edu quality classifier. Rules are simplified versions of the published ones."""
import gzip, hashlib, re, collections, json
import numpy as np
from common import Log, save

log = Log('ch4_data')
R = {}


def read_wet(path):
    """Yield (url, language header, text) for every 'conversion' record of a WET file (stops cleanly at a cut-off end)."""
    raw = b''
    try:
        with gzip.open(path) as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    break
                raw += chunk
    except (EOFError, OSError):
        pass                                           # the byte-range download ends mid-record
    for rec in raw.decode('utf-8', 'replace').split('WARC/1.0\r\n')[1:]:
        head, _, body = rec.partition('\r\n\r\n')
        if 'WARC-Type: conversion' not in head:
            continue
        url = re.search(r'WARC-Target-URI: (\S+)', head)
        lang = re.search(r'WARC-Identified-Content-Language: (\S+)', head)
        yield (url.group(1) if url else '', lang.group(1) if lang else '', body.strip())


docs = list(read_wet('/Users/admin/.cache/ccwet/part.wet.gz'))[:-1]     # drop the last, cut-off record
log(f'raw documents: {len(docs):,}   characters: {sum(len(d[2]) for d in docs):,}')
counts = collections.Counter(d[1].split(',')[0] for d in docs)
log('top languages (Common Crawl header, CLD2):', ', '.join(f'{k or "none"} {v}' for k, v in counts.most_common(8)))
R['raw'] = len(docs); R['langs'] = counts.most_common(10)

STOP = {'the', 'be', 'to', 'of', 'and', 'that', 'have', 'with'}


def gopher_quality(t):
    words = t.split()
    n = len(words)
    if not 50 <= n <= 100_000: return 'word count'
    mwl = sum(len(w) for w in words) / n
    if not 3 <= mwl <= 10: return 'mean word length'
    if (t.count('#') + t.count('...') + t.count('…')) / n > 0.1: return 'symbol ratio'
    lines = [l for l in t.split('\n') if l.strip()]
    if sum(l.lstrip().startswith(('•', '-', '*')) for l in lines) / len(lines) > 0.9: return 'bullet lines'
    if sum(l.rstrip().endswith(('...', '…')) for l in lines) / len(lines) > 0.3: return 'ellipsis lines'
    if sum(bool(re.search('[a-zA-Z]', w)) for w in words) / n < 0.8: return 'non-alphabetic words'
    if len(STOP & {w.lower() for w in words}) < 2: return 'stop words'
    return None


def gopher_repetition(t):
    lines = [l for l in t.split('\n') if l.strip()]
    c = collections.Counter(lines)
    dup_lines = sum(v for v in c.values() if v > 1)
    if dup_lines / len(lines) > 0.3: return 'duplicate lines'
    words = t.split()
    bigrams = collections.Counter(zip(words, words[1:]))
    if bigrams:
        (a, b), k = bigrams.most_common(1)[0]
        if k * (len(a) + len(b)) / max(1, sum(len(w) for w in words)) > 0.2: return 'top 2-gram'
    return None


def c4_rules(t):
    low = t.lower()
    if 'lorem ipsum' in low: return 'lorem ipsum', t
    if '{' in t: return 'curly bracket', t
    keep = [l for l in t.split('\n') if len(l.split()) >= 3 and 'javascript' not in l.lower()
            and not any(p in l.lower() for p in ['terms of use', 'privacy policy', 'cookie policy', 'uses cookies'])]
    t2 = '\n'.join(keep)
    if len(re.findall(r'[.!?]', t2)) < 5: return 'too few sentences', t2
    return None, t2


def fineweb_rules(t):
    lines = [l for l in t.split('\n') if l.strip()]
    if sum(l.rstrip().endswith(('.', '!', '?', '"', "'")) for l in lines) / len(lines) <= 0.12: return 'lines ending in punctuation <= 0.12'
    c = collections.Counter(lines)
    if sum(len(l) * v for l, v in c.items() if v > 1) / max(1, len(t)) >= 0.1: return 'chars in duplicated lines >= 0.1'
    if sum(len(l) < 30 for l in lines) / len(lines) >= 0.67: return 'short lines >= 0.67'
    return None


stages = collections.OrderedDict()
reasons = collections.Counter()
keep = []
for url, lang, t in docs:
    if not lang.startswith('eng'):
        reasons['not English'] += 1; continue
    r = gopher_quality(t) or gopher_repetition(t)
    if r:
        reasons['Gopher: ' + r] += 1; continue
    r, t = c4_rules(t)
    if r:
        reasons['C4: ' + r] += 1; continue
    r = fineweb_rules(t)
    if r:
        reasons['FineWeb: ' + r] += 1; continue
    keep.append((url, t))
log('')
log('why documents were removed:')
for k, v in reasons.most_common():
    log(f'   {k:45s} {v:5d}')
eng = sum(1 for d in docs if d[1].startswith('eng'))
g = eng - sum(v for k, v in reasons.items() if k.startswith('Gopher'))
c4 = g - sum(v for k, v in reasons.items() if k.startswith('C4'))
funnel = [('raw WET records', len(docs)), ('English', eng), ('Gopher rules', g), ('C4 rules', c4), ('FineWeb rules', len(keep))]
log('funnel: ' + ' -> '.join(f'{k} {v}' for k, v in funnel))
R['reasons'], R['funnel'] = reasons.most_common(), funnel
R['chars_raw'] = sum(len(d[2]) for d in docs); R['chars_keep'] = sum(len(t) for _, t in keep)

# ---- MinHash near-duplicate detection (FineWeb: word 5-grams, 112 hashes = 14 bands x 8 rows)
P = (1 << 61) - 1
rng = np.random.default_rng(0)
A = rng.integers(1, P, 112, dtype=np.uint64); Bc = rng.integers(0, P, 112, dtype=np.uint64)


def shingles(t, n=5):
    w = re.findall(r'\w+', t.lower())
    return {' '.join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))}


def minhash(sh):
    x = np.array([int.from_bytes(hashlib.sha1(s.encode()).digest()[:4], 'little') for s in sh], dtype=np.uint64)
    # (a*x + b) mod P for 112 hash functions at once, done in Python ints to avoid overflow
    return np.array([min(((int(a) * int(v) + int(b)) % P) for v in x) for a, b in zip(A, Bc)], dtype=np.uint64)


sigs, shs = [], []
for _, t in keep:
    s = shingles(t); shs.append(s); sigs.append(minhash(s))
buckets = collections.defaultdict(list)
for i, sg in enumerate(sigs):
    for b in range(14):
        buckets[(b, sg[b * 8:(b + 1) * 8].tobytes())].append(i)
pairs = {tuple(sorted((v[0], u))) for v in buckets.values() if len(v) > 1 for u in v[1:]}
log('')
log(f'MinHash on {len(keep)} kept documents: {len(pairs)} candidate duplicate pairs')
R['dups'] = []
for i, j in sorted(pairs)[:8]:
    jac = len(shs[i] & shs[j]) / len(shs[i] | shs[j])
    est = float(np.mean(sigs[i] == sigs[j]))
    log(f'   {keep[i][0][:60]:60s} ~ {keep[j][0][:60]:60s} true Jaccard {jac:.2f}, MinHash estimate {est:.2f}')
    R['dups'].append({'a': keep[i][0], 'b': keep[j][0], 'jac': jac, 'est': est})
dup_ids = {j for _, j in pairs}
keep2 = [k for i, k in enumerate(keep) if i not in dup_ids]
log(f'after removing one of each pair: {len(keep2)} documents')
R['after_dedup'] = len(keep2)
R['band_curve'] = [[s / 100, 1 - (1 - (s / 100) ** 8) ** 14] for s in range(0, 101, 2)]
for s in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
    log(f'   P(caught | Jaccard {s}) = 1 - (1 - {s}^8)^14 = {1 - (1 - s ** 8) ** 14:.3f}')

# ---- the FineWeb-Edu classifier (a regression head on Snowflake-arctic-embed-m, output 0..5)
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
name = 'HuggingFaceFW/fineweb-edu-classifier'
tk = AutoTokenizer.from_pretrained(name)
clf = AutoModelForSequenceClassification.from_pretrained(name).to('mps').eval()
scores = []
with torch.no_grad():
    for i in range(0, len(keep2), 16):
        enc = tk([t for _, t in keep2[i:i + 16]], return_tensors='pt', padding='longest', truncation=True, max_length=512).to('mps')
        scores += clf(**enc).logits.squeeze(-1).float().cpu().tolist()
ints = [int(round(max(0, min(s, 5)))) for s in scores]
hist = collections.Counter(ints)
log('')
log('FineWeb-Edu classifier scores (rounded 0..5): ' + ', '.join(f'{k}: {hist.get(k, 0)}' for k in range(6)))
log(f'kept by FineWeb-Edu (score >= 3): {sum(i >= 3 for i in ints)} of {len(ints)} ({sum(i >= 3 for i in ints) / len(ints):.1%})')
order = np.argsort(scores)
R['edu_hist'] = [hist.get(k, 0) for k in range(6)]
R['edu_examples'] = []
for i in list(order[:3]) + list(order[-3:]):
    snippet = re.sub(r'\s+', ' ', keep2[i][1])[:200]
    if scores[i] < 0.5:                       # the lowest scores here are adult pages: show the score only, not the text
        log(f'   score {scores[i]:.2f}  (adult site, text not shown)')
        R['edu_examples'].append({'score': scores[i], 'url': '(adult site)', 'text': ''})
        continue
    log(f'   score {scores[i]:.2f}  {keep2[i][0][:70]}')
    log(f'       "{snippet[:150]}"')
    R['edu_examples'].append({'score': scores[i], 'url': keep2[i][0], 'text': snippet})
# one raw document that the rules removed, for the text
R['removed_examples'] = []
for url, lang, t in docs[:400]:
    if lang.startswith('eng'):
        r = gopher_quality(t) or gopher_repetition(t)
        if r and len(R['removed_examples']) < 4:
            R['removed_examples'].append({'url': url, 'reason': r, 'text': re.sub(r'\s+', ' ', t)[:220]})
save('ch4_data', R)
