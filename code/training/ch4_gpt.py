"""A tiny GPT (decoder-only Transformer) for the Chapter 4 pretraining run, plus the data helpers.
GPT-2 style: token + position embeddings, N pre-norm blocks (attention then MLP), a final LayerNorm,
and an output layer that shares its weights with the token embedding."""
import math, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = os.path.expanduser('~/.cache/tinystories')


class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)              # queries, keys and values in one matrix
        self.proj = nn.Linear(d, d)                 # mixes the heads back together
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.n_head = n_head

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (t.view(B, T, self.n_head, d // self.n_head).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # each position sees only the past
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))          # residual connection 1
        x = x + self.mlp(self.ln2(x))                                  # residual connection 2
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d=384, n_layer=6, n_head=6, ctx=256):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight          # weight tying: one matrix for input and output
        self.ctx = ctx
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def n_params(self, non_embedding=True):
        n = sum(p.numel() for p in self.parameters())
        return n - self.tok.weight.numel() - self.pos.weight.numel() if non_embedding else n

    @torch.no_grad()
    def generate(self, idx, n, temperature=0.8, top_k=40, eos=None):
        for _ in range(n):
            logits = self(idx[:, -self.ctx:])[:, -1, :] / temperature
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if eos is not None and nxt.item() == eos:
                break
        return idx


def lr_at(step, peak, warmup, total, floor=0.1):
    """Linear warmup to `peak`, then cosine decay down to floor * peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def get_tokenizer(vocab=4096):
    """Train (once) a byte-level BPE tokenizer on the TinyStories text and cache it."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    path = os.path.join(DATA, f'bpe{vocab}.json')
    if os.path.exists(path):
        return Tokenizer.from_file(path)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=['<|endoftext|>'],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train([os.path.join(DATA, 'TinyStoriesV2-GPT4-train-head60MB.txt')], tr)
    tok.save(path)
    return tok


def get_tokens(split, tok):
    """Encode a text file once into a flat uint16 array of token ids, stories separated by <|endoftext|>."""
    name = {'train': 'TinyStoriesV2-GPT4-train-head60MB.txt', 'val': 'TinyStoriesV2-GPT4-valid.txt'}[split]
    path = os.path.join(DATA, f'{split}_{tok.get_vocab_size()}.npy')
    if os.path.exists(path):
        return np.load(path)
    eot = tok.token_to_id('<|endoftext|>')
    stories = [s.strip() for s in open(os.path.join(DATA, name)).read().split('<|endoftext|>') if s.strip()]
    if split == 'train':
        stories = stories[1:-1]                  # the byte-range download cuts the first and last story
    ids = []
    for enc in tok.encode_batch(stories):
        ids += enc.ids + [eot]
    arr = np.array(ids, dtype=np.uint16)
    np.save(path, arr)
    return arr


def batch(data, B, T, gen, device):
    """B random windows of T+1 tokens: x is the window, y is the same window shifted left by one."""
    ix = torch.randint(len(data) - T - 1, (B,), generator=gen)
    x = torch.stack([torch.from_numpy(data[i:i + T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + T + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)
