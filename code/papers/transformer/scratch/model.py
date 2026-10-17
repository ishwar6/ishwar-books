"""The Transformer of "Attention Is All You Need" (Vaswani et al., 2017), written from scratch in PyTorch.

One small class per block of Figure 1. We use only basic torch.nn parts (nn.Linear, nn.Embedding, nn.LayerNorm,
nn.Dropout). We do NOT use nn.MultiheadAttention or nn.Transformer: every step of attention is written out here.

Shapes in the comments use these names:
    b     batch size
    n     source length (tokens in the input sentence)
    t     target length (tokens the decoder has seen so far)
    d_model  width of every token vector (512 in the paper's base model)
    h     number of attention heads (8 in the paper)
    d_k   width of one head = d_model / h (64 in the paper)
    d_ff  width of the inner feed-forward layer (2048 in the paper)

Masks are boolean tensors: True means "this position may be looked at", False means "hide it".

The forward functions have no Python `if` that depends on the values inside a tensor, so the model exports to ONNX.
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 37000          # one vocabulary shared by source and target (paper §5.1: about 37,000 tokens)
    d_model: int = 512               # width of every token vector
    n_heads: int = 8                 # h, the number of attention heads
    d_ff: int = 2048                 # inner width of the feed-forward network
    n_encoder_layers: int = 6        # N = 6 in the paper
    n_decoder_layers: int = 6
    dropout: float = 0.1             # P_drop = 0.1 (paper §5.4)
    activation: str = 'relu'         # 'relu' (paper) or 'swish' (Marian)
    pos_layout: str = 'interleaved'  # 'interleaved' (paper: sin, cos, sin, cos, ...) or 'sin_then_cos' (Marian)
    max_len: int = 512               # longest sequence we prepare position vectors for
    pad_id: int = 0                  # id of the padding token
    bias: bool = True                # the paper's formulas have no bias in W^Q, W^K, W^V, W^O; Marian has biases
    final_logits_bias: bool = False  # Marian adds one learned number per vocabulary entry to the logits


# --------------------------------------------------------------------------------------------------------------------
# Embeddings (§3.4) and positional encoding (§3.5)
# --------------------------------------------------------------------------------------------------------------------

class TokenEmbedding(nn.Module):
    """Look up one learned vector per token, then multiply it by sqrt(d_model) (§3.4)."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model)  # (vocab_size, d_model): one row per token
        self.scale = math.sqrt(d_model)                 # 22.6 for d_model = 512
        # Small start values (std 1/sqrt(d_model)), so that after the scale each number has std about 1.
        nn.init.normal_(self.table.weight, mean=0.0, std=d_model ** -0.5)

    def forward(self, ids):                     # ids: (b, n) integers
        return self.table(ids) * self.scale     # (b, n, d_model)


def sinusoid_table(max_len, d_model, layout='interleaved'):
    """The fixed position vectors of §3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    'interleaved' puts them where the paper says: sin in even columns, cos in odd columns.
    'sin_then_cos' puts all the sines first and all the cosines after them. Marian (and tensor2tensor) do this.
    It is the same set of numbers in a different column order.
    """
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)        # (max_len, 1): 0, 1, 2, ...
    i2 = torch.arange(0, d_model, 2, dtype=torch.float32)                # (d_model/2,): 0, 2, 4, ... (this is 2i)
    freq = 1.0 / (10000.0 ** (i2 / d_model))                             # (d_model/2,): from 1 down to about 1/10000
    angle = pos * freq                                                   # (max_len, d_model/2)
    pe = torch.zeros(max_len, d_model)
    if layout == 'interleaved':
        pe[:, 0::2] = torch.sin(angle)                                   # even columns 0, 2, 4, ...
        pe[:, 1::2] = torch.cos(angle)                                   # odd columns 1, 3, 5, ...
    elif layout == 'sin_then_cos':
        half = d_model // 2
        pe[:, :half] = torch.sin(angle)                                  # first half: all the sines
        pe[:, half:] = torch.cos(angle)                                  # second half: all the cosines
    else:
        raise ValueError(layout)
    return pe                                                            # (max_len, d_model)


class PositionalEncoding(nn.Module):
    """Add the fixed sinusoid vector of each position to the token vectors (§3.5). Nothing here is learned."""

    def __init__(self, d_model, max_len=512, layout='interleaved'):
        super().__init__()
        # A buffer is saved with the model but is not a trained parameter.
        self.register_buffer('pe', sinusoid_table(max_len, d_model, layout), persistent=False)

    def forward(self, x):                       # x: (b, n, d_model)
        n = x.size(1)
        return x + self.pe[:n].unsqueeze(0)     # (1, n, d_model) is added to every sentence in the batch


# --------------------------------------------------------------------------------------------------------------------
# Attention (§3.2)
# --------------------------------------------------------------------------------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Equation 1: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Q: (..., n_q, d_k)   one query per position that asks
    K: (..., n_k, d_k)   one key per position that can be looked at
    V: (..., n_k, d_v)   one value per position: what it gives back
    mask: bool, broadcastable to (..., n_q, n_k); False = do not look there
    Returns the output (..., n_q, d_v) and the attention weights (..., n_q, n_k).
    """
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (..., n_q, n_k): how well each query matches each key
    if mask is not None:
        # A very negative score becomes 0 after softmax: the hidden positions get no weight.
        scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(scores, dim=-1)              # (..., n_q, n_k): each row sums to 1
    output = weights @ V                                 # (..., n_q, d_v): weighted average of the values
    return output, weights


class MultiHeadAttention(nn.Module):
    """§3.2.2: MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O, with head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).

    We keep the h small matrices W_i^Q (each d_model x d_k) side by side in one d_model x d_model matrix.
    This is the same maths: one big multiplication, then we cut the result into h pieces of width d_k.
    """

    def __init__(self, d_model, n_heads, bias=True, attn_dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0, 'd_model must be a multiple of the number of heads'
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=bias)   # all h query projections W_i^Q, side by side
        self.W_k = nn.Linear(d_model, d_model, bias=bias)   # all h key projections W_i^K
        self.W_v = nn.Linear(d_model, d_model, bias=bias)   # all h value projections W_i^V
        self.W_o = nn.Linear(d_model, d_model, bias=bias)   # W^O mixes the concatenated heads
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.last_weights = None                            # kept after each call, for drawing attention maps

    def split_heads(self, x):
        """(b, n, d_model) -> (b, h, n, d_k): cut each vector into h pieces, one per head."""
        b, n, _ = x.shape
        x = x.view(b, n, self.h, self.d_k)                  # (b, n, h, d_k)
        return x.transpose(1, 2)                            # (b, h, n, d_k): heads become a batch dimension

    def merge_heads(self, x):
        """(b, h, n, d_k) -> (b, n, d_model): put the h head outputs back side by side (the Concat)."""
        b, h, n, d_k = x.shape
        x = x.transpose(1, 2)                               # (b, n, h, d_k)
        return x.reshape(b, n, h * d_k)                     # (b, n, d_model)

    def forward(self, query, key, value, mask=None):
        # query: (b, n_q, d_model); key and value: (b, n_k, d_model)
        # In self-attention all three are the same tensor. In cross-attention key = value = encoder output.
        Q = self.split_heads(self.W_q(query))               # (b, h, n_q, d_k)
        K = self.split_heads(self.W_k(key))                 # (b, h, n_k, d_k)
        V = self.split_heads(self.W_v(value))               # (b, h, n_k, d_k)
        out, weights = scaled_dot_product_attention(Q, K, V, mask)   # out: (b, h, n_q, d_k)
        self.last_weights = weights                         # (b, h, n_q, n_k)
        out = self.merge_heads(out)                         # (b, n_q, d_model)
        return self.W_o(out)                                # (b, n_q, d_model)


# --------------------------------------------------------------------------------------------------------------------
# Feed-forward network (§3.3) and the residual + layer norm wrapper (§3.1)
# --------------------------------------------------------------------------------------------------------------------

class FeedForward(nn.Module):
    """§3.3: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2, applied to each position on its own."""

    def __init__(self, d_model, d_ff, activation='relu', dropout=0.0):
        super().__init__()
        self.W_1 = nn.Linear(d_model, d_ff)    # 512 -> 2048: widen
        self.W_2 = nn.Linear(d_ff, d_model)    # 2048 -> 512: back to the model width
        self.activation = activation
        self.dropout = nn.Dropout(dropout)     # 0 in the paper; some code drops here too

    def forward(self, x):                      # x: (b, n, d_model)
        hidden = self.W_1(x)                   # (b, n, d_ff)
        if self.activation == 'relu':          # this `if` reads a fixed setting, not tensor data: export is fine
            hidden = F.relu(hidden)            # max(0, .) as in the paper
        else:
            hidden = F.silu(hidden)            # swish(x) = x * sigmoid(x), used by Marian
        return self.W_2(self.dropout(hidden))  # (b, n, d_model)


class AddAndNorm(nn.Module):
    """§3.1 and §5.4: output = LayerNorm(x + Dropout(Sublayer(x))). This is "post-norm", as in the paper."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)      # eps 1e-5, the same as Marian and nn.TransformerEncoderLayer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_out):        # both: (b, n, d_model)
        return self.norm(x + self.dropout(sublayer_out))


# --------------------------------------------------------------------------------------------------------------------
# One encoder layer and one decoder layer (§3.1, Figure 1)
# --------------------------------------------------------------------------------------------------------------------

class EncoderLayer(nn.Module):
    """Self-attention, then feed-forward. Each one is wrapped in AddAndNorm."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.bias)
        self.add_norm_1 = AddAndNorm(cfg.d_model, cfg.dropout)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.activation)
        self.add_norm_2 = AddAndNorm(cfg.d_model, cfg.dropout)

    def forward(self, x, src_mask):            # x: (b, n, d_model); src_mask: (b, 1, 1, n)
        x = self.add_norm_1(x, self.self_attn(x, x, x, src_mask))   # every word looks at every (non-pad) word
        x = self.add_norm_2(x, self.ffn(x))                         # then each position is processed alone
        return x                                                    # (b, n, d_model)


class DecoderLayer(nn.Module):
    """Masked self-attention, then cross-attention to the encoder output, then feed-forward."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.bias)
        self.add_norm_1 = AddAndNorm(cfg.d_model, cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.bias)
        self.add_norm_2 = AddAndNorm(cfg.d_model, cfg.dropout)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.activation)
        self.add_norm_3 = AddAndNorm(cfg.d_model, cfg.dropout)

    def forward(self, y, memory, tgt_mask, src_mask):
        # y: (b, t, d_model) target so far; memory: (b, n, d_model) encoder output
        # tgt_mask: (1, 1, t, t) causal; src_mask: (b, 1, 1, n)
        y = self.add_norm_1(y, self.self_attn(y, y, y, tgt_mask))            # look only at earlier target words
        y = self.add_norm_2(y, self.cross_attn(y, memory, memory, src_mask)) # queries from y, keys/values from source
        y = self.add_norm_3(y, self.ffn(y))
        return y                                                             # (b, t, d_model)


# --------------------------------------------------------------------------------------------------------------------
# Masks (§3.2.3)
# --------------------------------------------------------------------------------------------------------------------

def padding_mask(ids, pad_id):
    """(b, n) ids -> (b, 1, 1, n) bool. False at padding, so no query looks at a pad token."""
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def causal_mask(t, device=None):
    """(1, 1, t, t) bool. Row i is True for columns 0..i: position i may look at itself and earlier positions only."""
    i = torch.arange(t, device=device).unsqueeze(1)    # (t, 1) query positions
    j = torch.arange(t, device=device).unsqueeze(0)    # (1, t) key positions
    return (j <= i).unsqueeze(0).unsqueeze(0)          # lower triangle, including the diagonal


# --------------------------------------------------------------------------------------------------------------------
# The two stacks and the whole model
# --------------------------------------------------------------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, cfg: Config, embedding: TokenEmbedding):
        super().__init__()
        self.embed = embedding                                  # shared with the decoder (§3.4)
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.pos_layout)
        self.dropout = nn.Dropout(cfg.dropout)                  # §5.4: dropout on embeddings + positions
        self.layers = nn.ModuleList([EncoderLayer(cfg) for _ in range(cfg.n_encoder_layers)])

    def forward(self, src_ids, src_mask):                       # src_ids: (b, n)
        x = self.dropout(self.pos(self.embed(src_ids)))         # (b, n, d_model)
        for layer in self.layers:                               # N layers, each with its own weights
            x = layer(x, src_mask)
        return x                                                # (b, n, d_model): the "memory"


class Decoder(nn.Module):
    def __init__(self, cfg: Config, embedding: TokenEmbedding):
        super().__init__()
        self.embed = embedding
        self.pos = PositionalEncoding(cfg.d_model, cfg.max_len, cfg.pos_layout)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_decoder_layers)])

    def forward(self, tgt_ids, memory, tgt_mask, src_mask):     # tgt_ids: (b, t)
        y = self.dropout(self.pos(self.embed(tgt_ids)))         # (b, t, d_model)
        for layer in self.layers:
            y = layer(y, memory, tgt_mask, src_mask)
        return y                                                # (b, t, d_model)


class Transformer(nn.Module):
    """The full encoder-decoder of Figure 1, with one embedding table used three times (§3.4):
    by the encoder input, by the decoder input, and (transposed) as the final linear layer before the softmax."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = TokenEmbedding(cfg.vocab_size, cfg.d_model)
        self.encoder = Encoder(cfg, self.embedding)
        self.decoder = Decoder(cfg, self.embedding)
        # Marian adds a fixed vector to the logits. The paper has no such term, so it is zero by default.
        self.register_buffer('logits_bias', torch.zeros(cfg.vocab_size), persistent=cfg.final_logits_bias)
        for name, p in self.named_parameters():                 # Xavier start values for every weight matrix
            if p.dim() == 2 and 'embedding' not in name:
                nn.init.xavier_uniform_(p)

    # The three steps, kept apart so that decoding and ONNX export can call them one by one.
    def encode(self, src_ids):
        """(b, n) ids -> memory (b, n, d_model) and the source padding mask (b, 1, 1, n)."""
        src_mask = padding_mask(src_ids, self.cfg.pad_id)
        return self.encoder(src_ids, src_mask), src_mask

    def decode(self, tgt_ids, memory, src_mask):
        """Target so far (b, t) -> logits (b, t, vocab_size). Row k predicts token k+1."""
        t = tgt_ids.size(1)
        # Only the causal mask. Targets are padded on the right, so a real token never sees a later pad anyway.
        # (And Marian starts the decoder with the pad id itself: a padding mask here would hide that start token.)
        tgt_mask = causal_mask(t, tgt_ids.device)               # (1, 1, t, t), broadcast over the batch
        y = self.decoder(tgt_ids, memory, tgt_mask, src_mask)   # (b, t, d_model)
        return self.generator(y)

    def generator(self, y):
        """The final Linear of Figure 1, tied to the embedding table: logits = y E^T (+ bias)."""
        return y @ self.embedding.table.weight.T + self.logits_bias   # (b, t, vocab_size)

    def forward(self, src_ids, tgt_ids):
        memory, src_mask = self.encode(src_ids)
        return self.decode(tgt_ids, memory, src_mask)           # (b, t, vocab_size); softmax is in the loss


def count_parameters(model):
    """Count each tensor once: the shared embedding table is counted once, not three times."""
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------------------------------------------------
# Making an output sentence: greedy decoding and beam search
# --------------------------------------------------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode(model, src_ids, bos_id, eos_id, max_len=64):
    """Encode once, then add the most probable next token, one step at a time, until every row has eos.
    src_ids: (b, n). Returns (b, <= max_len + 1) ids that start with bos_id."""
    model.eval()
    memory, src_mask = model.encode(src_ids)
    b = src_ids.size(0)
    out = torch.full((b, 1), bos_id, dtype=torch.long, device=src_ids.device)   # (b, 1): start token
    done = torch.zeros(b, dtype=torch.bool, device=src_ids.device)
    for _ in range(max_len):
        logits = model.decode(out, memory, src_mask)          # (b, t, vocab): we re-run the whole prefix
        next_id = logits[:, -1].argmax(-1)                    # (b,): best token after the last position
        next_id = torch.where(done, torch.full_like(next_id, model.cfg.pad_id), next_id)  # pad finished rows
        out = torch.cat([out, next_id.unsqueeze(1)], dim=1)
        done = done | (next_id == eos_id)
        if bool(done.all()):
            break
    return out


@torch.no_grad()
def beam_search(model, src_ids, bos_id, eos_id, beam_size=4, max_len=64, alpha=0.6):
    """Beam search for ONE sentence (src_ids: (1, n)), with the length penalty of Wu et al. (2016), as in §6.1.
    Keep the beam_size best partial outputs; each step, extend each by every token and keep the best again."""
    model.eval()
    memory, src_mask = model.encode(src_ids)                  # (1, n, d_model)
    beams = [([bos_id], 0.0)]                                 # (token list, sum of log-probabilities)
    finished = []
    for _ in range(max_len):
        seqs = torch.tensor([s for s, _ in beams], device=src_ids.device)     # (k, t)
        k = seqs.size(0)
        logp = torch.log_softmax(model.decode(seqs, memory.expand(k, -1, -1), src_mask.expand(k, -1, -1, -1))[:, -1], -1)
        cand = []
        for row, (seq, score) in enumerate(beams):
            top = torch.topk(logp[row], beam_size)
            for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                cand.append((seq + [tok], score + lp))
        cand.sort(key=lambda c: c[1], reverse=True)
        beams = []
        for seq, score in cand:
            if seq[-1] == eos_id:
                finished.append((seq, score))
            else:
                beams.append((seq, score))
            if len(beams) == beam_size:
                break
        if len(finished) >= beam_size or not beams:
            break
    finished = finished or beams

    def penalised(c):                                         # lp(Y) = ((5 + |Y|) / 6) ^ alpha
        seq, score = c
        return score / (((5 + len(seq)) / 6) ** alpha)
    return max(finished, key=penalised)[0]


def config_dict(cfg: Config):
    return asdict(cfg)
