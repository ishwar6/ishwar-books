"""Part 3: why a normal language model cannot simply look both ways ("each word would indirectly see itself").

Three tiny Transformers are trained on the same real text (WikiText-2) to predict tokens:
  A. left-to-right LM, 2 layers: predict token t+1 from tokens 1..t (a causal mask). The honest baseline.
  B. "both sides", 1 layer: predict token t from every OTHER token. Position t never sees its own token:
     the attention mask blocks it, and the query at t is built from the position alone, not the token.
  C. the same as B, but 2 layers.
B is a fair fill-in-the-blank model. In C, layer 1 at a neighbour position has already read token t,
so in layer 2 position t can read its own token back from the neighbour. The loss collapses: it cheats.
Seed fixed; runs on the Apple GPU (MPS) if there is one."""
import math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from transformers import BertTokenizer
from datasets import load_dataset
from bert_part3_data import detok
from common import Log, save

log = Log('part3_seeitself')
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
tok = BertTokenizer.from_pretrained('bert-base-uncased')
ds = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')


def ids_of(split):
    text = ' '.join(detok(l) for l in ds[split]['text'] if l.strip() and not l.strip().startswith('='))
    return torch.tensor(tok(text, add_special_tokens=False).input_ids)


train, valid = ids_of('train'), ids_of('validation')
# a small vocabulary: the 8,191 most common word pieces of the training text, everything else -> one "rare" id
counts = torch.bincount(train, minlength=tok.vocab_size)
keep = counts.argsort(descending=True)[:8191]
remap = torch.full((tok.vocab_size,), 8191, dtype=torch.long)
remap[keep] = torch.arange(8191)
train, valid = remap[train], remap[valid]
VOC, T, D, H = 8192, 64, 128, 4
log(f'text: WikiText-2, {len(train):,} training and {len(valid):,} validation word pieces; vocabulary {VOC:,}; windows of {T} tokens')


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, mask, query_from=None, residual=True):
        B, L, _ = x.shape
        h = self.ln1(x)
        qsrc = self.ln1(query_from) if query_from is not None else h
        split = lambda t: t.view(B, L, H, D // H).transpose(1, 2)
        q, k, v = split(self.q(qsrc)), split(self.k(h)), split(self.v(h))
        a = (q @ k.transpose(-1, -2)) / math.sqrt(D // H)
        a = a.masked_fill(~mask, float('-inf')).softmax(-1)
        att = self.o((a @ v).transpose(1, 2).reshape(B, L, D))
        x = (x + att) if residual else (query_from + att)
        return x + self.ff(self.ln2(x))


class Tiny(nn.Module):
    def __init__(self, layers, kind):
        super().__init__()
        self.kind = kind
        self.tok, self.pos = nn.Embedding(VOC, D), nn.Embedding(T, D)
        self.blocks = nn.ModuleList(Block() for _ in range(layers))
        self.out = nn.Linear(D, VOC)
        i = torch.arange(T)
        self.register_buffer('causal', (i[None, :] <= i[:, None]))       # row = query, column = key
        self.register_buffer('not_self', (i[None, :] != i[:, None]))

    def forward(self, ids):
        p = self.pos(torch.arange(ids.shape[1], device=ids.device))[None].expand(ids.shape[0], -1, -1)
        x = self.tok(ids) + p
        if self.kind == 'ltr':
            for b in self.blocks:
                x = b(x, self.causal)
        else:
            # layer 1: the query and the residual stream at position t hold only "position t", never token t
            x = self.blocks[0](x, self.not_self, query_from=p, residual=False)
            for b in self.blocks[1:]:
                x = b(x, self.not_self)
        return self.out(x)


def batch(data, n, g):
    s = torch.randint(0, len(data) - T - 1, (n,), generator=g)
    return torch.stack([data[i:i + T + 1] for i in s])


def loss_of(model, w):
    if model.kind == 'ltr':
        logits = model(w[:, :-1])
        return Fn.cross_entropy(logits.reshape(-1, VOC), w[:, 1:].reshape(-1))
    logits = model(w[:, :-1])
    return Fn.cross_entropy(logits.reshape(-1, VOC), w[:, :-1].reshape(-1))     # predict the token AT each position


STEPS, BS = 3000, 64
runs = {}
for name, layers, kind in (('A: left-to-right, 2 layers', 2, 'ltr'), ('B: both sides, 1 layer', 1, 'bi'), ('C: both sides, 2 layers', 2, 'bi')):
    torch.manual_seed(0)
    model = Tiny(layers, kind).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    curve, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        for pg in opt.param_groups:
            pg['lr'] = 1e-3 * min(1, step / 100)
        l = loss_of(model, batch(train, BS, g).to(dev))
        opt.zero_grad()
        l.backward()
        opt.step()
        if step % 50 == 0 or step == 1:
            curve.append((step, round(l.item(), 4)))
    model.eval()
    gv = torch.Generator().manual_seed(1)
    with torch.no_grad():
        vl = sum(loss_of(model, batch(valid, 64, gv).to(dev)).item() for _ in range(20)) / 20
    runs[name] = dict(layers=layers, kind=kind, curve=curve, train_last=curve[-1][1], valid=vl, seconds=round(time.time() - t0, 1))
    log(f'{name:28s} training loss: step 1 {curve[0][1]:.2f} -> step {STEPS} {curve[-1][1]:.2f}   validation loss {vl:.2f} '
        f'(perplexity {math.exp(vl):,.1f})   {time.time() - t0:.0f} s')

log('')
log(f'a guess spread evenly over the {VOC:,} ids would have loss ln({VOC}) = {math.log(VOC):.2f}')
save('part3_seeitself', dict(setup=dict(voc=VOC, window=T, d=D, heads=H, steps=STEPS, batch=BS, lr=1e-3, device=dev,
                                       train_tokens=len(train), valid_tokens=len(valid)), runs=runs, uniform=math.log(VOC)))
