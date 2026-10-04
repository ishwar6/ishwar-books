"""Part 5: a smaller, honest re-run of the paper's feature-based NER experiment (Section 5.3, Table 7).

BERT stays frozen. We read out its hidden states (all 13: the embedding output plus 12 layers) for every word of
CoNLL-2003, keep the vector of each word's FIRST sub-token (as the paper does), and train a small classifier
on top for six different choices of features. Then we score entity-level F1 on the CoNLL-2003 dev set.

Differences from the paper (so the numbers are NOT comparable one to one):
  - bert-base-cased, single sentences (no "maximal document context"),
  - the first 5,000 training sentences only (the full set has 14,041),
  - the classifier is a 2-layer BiLSTM with 384 units per direction (768 in total), trained 4 epochs, 3 seeds averaged,
  - no hyperparameter search.
"""
import random, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from common import Log, save

log = Log('part5_ner')
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
N_TRAIN = 5000
SEEDS = [0, 1, 2]
EPOCHS = 4

ds = load_dataset('eriktks/conll2003', revision='refs/convert/parquet')
names = ds['train'].features['ner_tags'].feature.names           # O, B-PER, I-PER, B-ORG, ...
keep = lambda split: [x for x in ds[split] if x['tokens'] != ['-DOCSTART-'] and len(x['tokens'])]
train, valid = keep('train')[:N_TRAIN], keep('validation')
log(f'CoNLL-2003: {len(train)} training sentences used, {len(valid)} dev sentences; labels: {names}')

tok = AutoTokenizer.from_pretrained('bert-base-cased')
bert = AutoModel.from_pretrained('bert-base-cased').to(dev).eval()


@torch.no_grad()
def features(sents, bs=32):
    """For every sentence: a (words, 13, 768) float16 tensor, the first sub-token of each word at every layer."""
    out = []
    for i in range(0, len(sents), bs):
        batch = sents[i:i + bs]
        enc = tok([s['tokens'] for s in batch], is_split_into_words=True, return_tensors='pt', padding=True, truncation=True, max_length=512)
        hs = torch.stack(bert(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True).hidden_states, 2)  # (B, T, 13, 768)
        for b, s in enumerate(batch):
            wid = enc.word_ids(b)
            first = [t for t, w in enumerate(wid) if w is not None and (t == 0 or wid[t - 1] != w)]
            assert len(first) == len(s['tokens'])
            out.append(hs[b, first].half().cpu())
    return out


t0 = time.time()
Ftr, Fdv = features(train), features(valid)
Ytr = [torch.tensor(s['ner_tags']) for s in train]
Ydv = [s['ner_tags'] for s in valid]
log(f'features: {sum(len(f) for f in Ftr)} train words, {sum(len(f) for f in Fdv)} dev words, 13 layers x 768 numbers each, {time.time() - t0:.0f} s')

# hidden_states index: 0 = embeddings, 1..12 = Transformer layers
CHOICES = {
    'Embeddings': ('pick', [0]),
    'Second-to-last hidden': ('pick', [11]),
    'Last hidden': ('pick', [12]),
    'Weighted sum last four': ('wsum', [9, 10, 11, 12]),
    'Concat last four': ('cat', [9, 10, 11, 12]),
    'Weighted sum all 12 layers': ('wsum', list(range(1, 13))),
}


class Tagger(torch.nn.Module):
    def __init__(self, mode, layers, n_labels):
        super().__init__()
        self.mode, self.layers = mode, layers
        d = 768 * len(layers) if mode == 'cat' else 768
        self.w = torch.nn.Parameter(torch.zeros(len(layers)))      # weights of the weighted sum (softmax of these)
        self.lstm = torch.nn.LSTM(d, 384, num_layers=2, bidirectional=True, batch_first=True)
        self.out = torch.nn.Linear(768, n_labels)

    def forward(self, x):                                           # x: (B, words, 13, 768)
        x = x[:, :, self.layers].float()
        if self.mode == 'wsum':
            x = (torch.softmax(self.w, 0)[None, None, :, None] * x).sum(2)
        else:
            x = x.flatten(2)
        return self.out(self.lstm(x)[0])


def pad(fs, ys=None):
    T = max(len(f) for f in fs)
    X = torch.zeros(len(fs), T, 13, 768, dtype=torch.float16)
    Y = torch.full((len(fs), T), -100)
    for i, f in enumerate(fs):
        X[i, :len(f)] = f
        if ys is not None:
            Y[i, :len(f)] = ys[i]
    return X, Y


def spans(tags):
    """Entities as (type, start, end) from BIO tags, conlleval style (an I- without a B- starts a new entity)."""
    out, cur = set(), None
    for i, t in enumerate(tags + ['O']):
        if t == 'O' or t.startswith('B-') or (t.startswith('I-') and (cur is None or cur[0] != t[2:])):
            if cur:
                out.add((cur[0], cur[1], i))
            cur = (t[2:], i) if t != 'O' else None
    return out


def f1(gold, pred):
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        G, P = spans([names[x] for x in g]), spans([names[x] for x in p])
        tp += len(G & P); fp += len(P - G); fn += len(G - P)
    prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
    return 100 * 2 * prec * rec / max(1e-9, prec + rec)


def run(name, seed):
    mode, layers = CHOICES[name]
    torch.manual_seed(seed); random.seed(seed)
    m = Tagger(mode, layers, len(names)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    order = list(range(len(Ftr)))
    for _ in range(EPOCHS):
        random.shuffle(order)
        for i in range(0, len(order), 32):
            idx = order[i:i + 32]
            X, Y = pad([Ftr[j] for j in idx], [Ytr[j] for j in idx])
            loss = torch.nn.functional.cross_entropy(m(X.to(dev)).flatten(0, 1), Y.to(dev).flatten(), ignore_index=-100)
            opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); pred = []
    with torch.no_grad():
        for i in range(0, len(Fdv), 64):
            X, _ = pad(Fdv[i:i + 64])
            P = m(X.to(dev)).argmax(-1).cpu()
            pred += [P[k, :len(f)].tolist() for k, f in enumerate(Fdv[i:i + 64])]
    return f1(Ydv, pred)


results = {}
for name in CHOICES:
    t0 = time.time()
    scores = [run(name, s) for s in SEEDS]
    results[name] = dict(scores=[round(x, 2) for x in scores], mean=round(float(np.mean(scores)), 2))
    log(f'{name:28s} dev F1 = {np.mean(scores):5.2f}   (seeds: {", ".join(f"{x:.2f}" for x in scores)}; {time.time() - t0:.0f} s)')

save('part5_ner', dict(n_train_sentences=len(train), n_dev_sentences=len(valid), seeds=SEEDS, epochs=EPOCHS,
                       classifier='2-layer BiLSTM, 384 units per direction, Adam lr 1e-3, batch 32', model='bert-base-cased (frozen)',
                       results=results))
