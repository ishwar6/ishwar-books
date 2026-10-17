"""Part 6 (results, variations, impact), the code walk-through. Each tool of Section 6 twice: from scratch, and with
the library, with a check that both agree. Every number of the code sections of part6.md comes from this output.

1. BLEU from scratch (n-gram precision, clipping, brevity penalty, corpus totals), checked against sacrebleu.
   sacrebleu is not in the venv; we install it into a scratch folder and run this script with PYTHONPATH.
2. Greedy decoding written around the released Marian model, against model.generate(num_beams=1).
3. Beam search with a length penalty written from scratch, against model.generate(num_beams=4, length_penalty=0.6);
   then the same search with the GNMT length penalty of the paper.
4. Checkpoint averaging on a tiny model, by hand and with torch.optim.swa_utils.AveragedModel.
5. The parameter counts of Table 3 from the shapes, against PyTorch layers built with the same shapes, and against
   the released Marian model.
6. The attention weights behind the heatmaps: encoder layer 5 of Marian by hand, against output_attentions; then the
   weights of the word "making" drawn as a text heatmap.
"""
import math
from collections import Counter
import torch
import torch.nn as nn
from common import Log, save

log = Log('part6_code')
R = {}

# ================================================================== 1. BLEU
log('== 1. BLEU from scratch')


def ngrams(words, n):
    """Count every group of n words that follow each other."""
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def bleu_stats(hyp, ref, max_n=4):
    """For one sentence: matches and totals for n = 1..4, and the two lengths."""
    hyp, ref = hyp.split(), ref.split()
    matches, totals = [], []
    for n in range(1, max_n + 1):
        h, r = ngrams(hyp, n), ngrams(ref, n)
        # clipping: a group counts at most as often as it occurs in the reference
        matches.append(sum(min(count, r[g]) for g, count in h.items()))
        totals.append(max(len(hyp) - n + 1, 0))
    return matches, totals, len(hyp), len(ref)


def bleu_from_stats(matches, totals, c, r):
    if min(matches) == 0:
        return 0.0                                   # log(0): one empty precision makes the score 0
    log_mean = sum(math.log(m / t) for m, t in zip(matches, totals)) / len(matches)   # geometric mean, in logs
    bp = 1.0 if c > r else math.exp(1 - r / c)       # brevity penalty: only short outputs lose points
    return 100 * bp * math.exp(log_mean)


def corpus_bleu(hyps, refs):
    """Corpus BLEU adds up the counts of all sentences first, then computes one score."""
    M, T, C, Rl = [0] * 4, [0] * 4, 0, 0
    for h, r in zip(hyps, refs):
        m, t, c, rl = bleu_stats(h, r)
        M = [a + b for a, b in zip(M, m)]
        T = [a + b for a, b in zip(T, t)]
        C, Rl = C + c, Rl + rl
    return bleu_from_stats(M, T, C, Rl)


try:
    import sacrebleu
except ImportError:
    sacrebleu = None
log('sacrebleu:', sacrebleu.__version__ if sacrebleu else 'not installed')

REF = 'the cat sat on the mat'
for hyp in ['the cat sat on a mat', 'the cat sat on the mat', 'the cat sat on', 'the the the the the the']:
    m, t, c, r = bleu_stats(hyp, REF)
    ours = bleu_from_stats(m, t, c, r)
    p = ' '.join(f'{a}/{b}' for a, b in zip(m, t))
    lib = f'{sacrebleu.sentence_bleu(hyp, [REF], smooth_method="none").score:.2f}' if sacrebleu else 'n/a'
    log(f'"{hyp}": p1..p4 = {p}, c = {c}, r = {r}, BLEU ours {ours:.2f}, sacrebleu {lib}')
if sacrebleu:   # sentence_bleu smooths empty precisions by default, so a hopeless sentence does not get exactly 0
    log(f'sacrebleu with its default smoothing on the last one: {sacrebleu.sentence_bleu(hyp, [REF]).score:.2f}')

# clipping, the long way round, for the last example
h = ngrams('the the the the the the'.split(), 1)
log(f'unclipped: "the" counted {h[("the",)]} times; clipped: min(6, {ngrams(REF.split(), 1)[("the",)]}) = '
    f'{min(6, ngrams(REF.split(), 1)[("the",)])}')

HYPS = ['the cat sat on a mat', 'a dog ran in the big park today']
REFS = ['the cat sat on the mat', 'the dog ran in the park today']
sent = [bleu_from_stats(*bleu_stats(hh, rr)) for hh, rr in zip(HYPS, REFS)]
ours = corpus_bleu(HYPS, REFS)
lib = f'{sacrebleu.corpus_bleu(HYPS, [REFS]).score:.2f}' if sacrebleu else 'n/a'
log(f'two sentences: sentence BLEU {sent[0]:.2f} and {sent[1]:.2f} (mean {sum(sent) / 2:.2f}); '
    f'corpus BLEU ours {ours:.2f}, sacrebleu {lib}')
R['bleu_toy'] = bleu_from_stats(*bleu_stats('the cat sat on a mat', REF))

# ================================================================== 2. greedy decoding
log('\n== 2. Greedy decoding by hand around Marian, against model.generate')
from transformers import MarianMTModel, MarianTokenizer
from transformers.modeling_outputs import BaseModelOutput
NAME = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME, attn_implementation='eager').eval()
START = model.config.decoder_start_token_id      # 58100, the decoder's first input
EOS = model.config.eos_token_id                  # 0, the end of a sentence
PAD = model.config.pad_token_id                  # 58100; Marian never writes it (bad_words_ids)


@torch.no_grad()
def encode(text):
    x = tok([text], return_tensors='pt')
    memory = model.get_encoder()(**x).last_hidden_state     # (1, n, 512), computed once
    return x, memory


@torch.no_grad()
def next_log_probs(x, memory, prefixes):
    """log P(next token | source, prefix) for each prefix. prefixes: (beams, t) -> (beams, V)."""
    b = prefixes.shape[0]
    out = model(encoder_outputs=BaseModelOutput(last_hidden_state=memory.expand(b, -1, -1)),
                attention_mask=x['attention_mask'].expand(b, -1), decoder_input_ids=prefixes)
    logits = out.logits[:, -1]                    # scores for the token after the prefix
    logits[:, PAD] = -float('inf')                # the padding token is never a word
    return torch.log_softmax(logits, dim=-1)


def greedy_decode(text, extra=50):
    x, memory = encode(text)
    ys = [START]
    for _ in range(x['input_ids'].shape[1] + extra):           # the paper's limit: input length + 50
        lp = next_log_probs(x, memory, torch.tensor([ys]))[0]
        best = int(lp.argmax())                                # the single best next token
        ys.append(best)
        if best == EOS:
            break
    return ys


SENTS = ['The law will never be perfect, but its application should be just.',
         'We trained the model on eight GPUs for twelve hours, and it still beat the ensembles.']
for s in SENTS:
    ours = greedy_decode(s)
    x = tok([s], return_tensors='pt')
    with torch.no_grad():
        lib = model.generate(**x, num_beams=1, do_sample=False, max_new_tokens=x['input_ids'].shape[1] + 50)[0].tolist()
    log(f'{s}\n  ours   : {tok.decode(ours, skip_special_tokens=True)}\n  library: same token ids: {ours == lib}')

# ================================================================== 3. beam search with a length penalty
log('\n== 3. Beam search with a length penalty by hand, against model.generate(num_beams=4, length_penalty=0.6)')


def lp_hf(n, alpha):
    return n ** alpha                             # the library: length to the power alpha


def lp_gnmt(n, alpha):
    return ((5 + n) / 6) ** alpha                 # the paper (from GNMT): ((5 + |Y|) / 6) ^ alpha


def beam_search(text, k=4, alpha=0.6, lp=lp_hf, extra=50):
    x, memory = encode(text)
    max_new = x['input_ids'].shape[1] + extra
    running = [([START], 0.0)]                    # (tokens, sum of log probabilities)
    finished = []                                 # (tokens, score after the length penalty)
    for t in range(1, max_new + 1):
        lp_next = next_log_probs(x, memory, torch.tensor([ys for ys, _ in running]))   # (beams, V)
        total = torch.tensor([s for _, s in running])[:, None] + lp_next               # score of every extension
        top_scores, top_idx = total.flatten().topk(2 * k)     # keep 2k, in case some of them end here
        V = lp_next.shape[1]
        new_running = []
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist())):
            beam, token = divmod(idx, V)
            ys = running[beam][0] + [token]
            if token == EOS:
                if rank < k:                      # only a top-k candidate may finish
                    finished.append((ys, score / lp(t, alpha)))
            elif len(new_running) < k:
                new_running.append((ys, score))
        finished = sorted(finished, key=lambda f: -f[1])[:k]
        running = new_running
        # stop when k sentences are finished and the best open one cannot beat the worst finished one
        if len(finished) == k and running[0][1] / lp(t, alpha) <= finished[-1][1]:
            break
    return finished[0]


for s in SENTS:
    ys, score = beam_search(s)
    x = tok([s], return_tensors='pt')
    with torch.no_grad():
        g = model.generate(**x, num_beams=4, length_penalty=0.6, max_new_tokens=x['input_ids'].shape[1] + 50,
                           return_dict_in_generate=True, output_scores=True)
    lib = g.sequences[0].tolist()
    lib = lib[:lib.index(EOS) + 1]
    log(f'{s}\n  ours   : {tok.decode(ys, skip_special_tokens=True)}  (score {score:.4f})')
    log(f'  library: same token ids: {ys == lib}, score {g.sequences_scores[0].item():.4f}')
    ys2, score2 = beam_search(s, lp=lp_gnmt)
    log(f'  with the paper\'s penalty ((5 + n) / 6)^0.6: {tok.decode(ys2, skip_special_tokens=True)}  (score {score2:.4f})')

# the length penalty on its own: the example of the article
for n, lpv in ((4, -2.0), (8, -2.4)):
    log(f'length {n}, log P {lpv}: GNMT factor {lp_gnmt(n, 0.6):.3f} -> {lpv / lp_gnmt(n, 0.6):.3f};  '
        f'library factor {lp_hf(n, 0.6):.3f} -> {lpv / lp_hf(n, 0.6):.3f}')

# ================================================================== 4. checkpoint averaging
log('\n== 4. Checkpoint averaging on a tiny model')
from torch.optim.swa_utils import AveragedModel

torch.manual_seed(0)
Xtr, Xva = torch.randn(512, 8), torch.randn(512, 8)
w_true = torch.randn(8, 1)
Ytr = torch.sin(Xtr @ w_true) + 0.3 * torch.randn(512, 1)    # a curve plus noise
Yva = torch.sin(Xva @ w_true) + 0.3 * torch.randn(512, 1)
net = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1))
opt = torch.optim.SGD(net.parameters(), lr=0.1)               # a large step, so the weights keep moving
checkpoints = []
for step in range(1, 3001):
    i = torch.randint(0, 512, (16,))
    loss = ((net(Xtr[i]) - Ytr[i]) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0:                                        # "written at 10-minute intervals"
        checkpoints.append({k: v.clone() for k, v in net.state_dict().items()})


def average_by_hand(state_dicts):
    """Weight by weight: the mean of the same tensor over all checkpoints."""
    return {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}


def val_loss(state):
    net.load_state_dict(state)
    with torch.no_grad():
        return ((net(Xva) - Yva) ** 2).mean().item()


last5 = checkpoints[-5:]
avg = average_by_hand(last5)
swa = AveragedModel(net)                  # Method 2: the library keeps a running mean
for sd in last5:
    net.load_state_dict(sd)
    swa.update_parameters(net)
diff = max((avg[k] - v).abs().max().item() for k, v in swa.module.state_dict().items())
log('one weight in the last 5 checkpoints: ' + ', '.join(f'{sd["0.weight"][0, 0].item():.4f}' for sd in last5) +
    f'  -> mean {avg["0.weight"][0, 0].item():.4f}')
log('validation loss of the last 5 checkpoints: ' + ', '.join(f'{val_loss(sd):.4f}' for sd in last5))
log(f'validation loss of their average: {val_loss(avg):.4f}')
log(f'average by hand vs AveragedModel: max abs difference {diff:.1e}')
R['avg'] = dict(ckpt=[val_loss(sd) for sd in last5], avg=val_loss(avg), diff=diff)

# ================================================================== 5. parameter counts from the shapes
log('\n== 5. Parameters of Table 3 rows: from the shapes, and from PyTorch layers with the same shapes')


def count_params(N=6, d_model=512, d_ff=2048, h=8, d_k=None, d_v=None, vocab=37000):
    d_k = d_k or d_model // h
    d_v = d_v or d_model // h
    attention = (d_model * h * d_k + h * d_k        # W_Q and its bias: d_model -> h heads of d_k
                 + d_model * h * d_k + h * d_k      # W_K
                 + d_model * h * d_v + h * d_v      # W_V
                 + h * d_v * d_model + d_model)     # W_O: the joined heads -> d_model
    ffn = d_model * d_ff + d_ff + d_ff * d_model + d_model   # two linear layers with biases
    norm = 2 * d_model                                        # LayerNorm: a gain and a bias
    encoder_layer = attention + ffn + 2 * norm                # self-attention, FFN, 2 norms
    decoder_layer = 2 * attention + ffn + 3 * norm            # + attention over the encoder, 3 norms
    embedding = vocab * d_model                               # one matrix, shared 3 ways (Part 2)
    return N * encoder_layer + N * decoder_layer + embedding


def count_with_torch(N=6, d_model=512, d_ff=2048, h=8, vocab=37000):
    """The same shapes built from PyTorch layers on the 'meta' device: no memory, only shapes."""
    with torch.device('meta'):
        enc = nn.ModuleList([nn.TransformerEncoderLayer(d_model, h, d_ff) for _ in range(N)])
        dec = nn.ModuleList([nn.TransformerDecoderLayer(d_model, h, d_ff) for _ in range(N)])
        emb = nn.Embedding(vocab, d_model)
    return sum(p.numel() for m in (enc, dec, emb) for p in m.parameters())


ROWS = [('base', {}), ('(A) h=1, d_k=512', dict(h=1, d_k=512, d_v=512)), ('(A) h=4, d_k=128', dict(h=4, d_k=128, d_v=128)),
        ('(A) h=16, d_k=32', dict(h=16, d_k=32, d_v=32)), ('(A) h=32, d_k=16', dict(h=32, d_k=16, d_v=16)),
        ('(B) d_k=16', dict(d_k=16)), ('(B) d_k=32', dict(d_k=32)), ('(C) N=2', dict(N=2)), ('(C) N=4', dict(N=4)),
        ('(C) N=8', dict(N=8)), ('(C) d_model=256', dict(d_model=256, d_k=32, d_v=32)),
        ('(C) d_model=1024', dict(d_model=1024, d_k=128, d_v=128)), ('(C) d_ff=1024', dict(d_ff=1024)),
        ('(C) d_ff=4096', dict(d_ff=4096)), ('big', dict(d_model=1024, d_ff=4096, h=16))]
PAPER = {'base': 65, '(B) d_k=16': 58, '(B) d_k=32': 60, '(C) N=2': 36, '(C) N=4': 50, '(C) N=8': 80,
         '(C) d_model=256': 28, '(C) d_model=1024': 168, '(C) d_ff=1024': 53, '(C) d_ff=4096': 90, 'big': 213}
log(f'{"row":20s} {"shapes (M)":>10s} {"PyTorch (M)":>11s} {"paper (M)":>9s}')
R['params'] = {}
for name, kw in ROWS:
    ours = count_params(**kw)
    d, hh = kw.get('d_model', 512), kw.get('h', 8)
    if kw.get('d_k', d // hh) * hh == d:          # nn.MultiheadAttention needs h x d_k = d_model
        lib = count_with_torch(**{k: v for k, v in kw.items() if k not in ('d_k', 'd_v')})
        lib_s = f'{lib / 1e6:11.2f}' + (' =' if lib == ours else ' !')
    else:
        lib_s = '   (cannot)'
    log(f'{name:20s} {ours / 1e6:10.2f} {lib_s} {str(PAPER.get(name, "-")):>9s}')
    R['params'][name] = ours

marian_count = sum(p.numel() for p in model.parameters())
ours = count_params(vocab=model.config.vocab_size)
pos = sum(p.numel() for n, p in model.named_parameters() if 'embed_positions' in n)
log(f'Marian: count_params(vocab=58,101) = {ours:,};  sum of the model\'s own tensors = {marian_count:,};  '
    f'difference {marian_count - ours:,}')
log(f'  the difference is the two stored tables of sinusoidal positions, 2 x 512 x 512 = {pos:,}')
R['marian'] = [ours, marian_count]

# ================================================================== 6. the attention weights behind the heatmaps
log('\n== 6. Attention weights of encoder layer 5: by hand and output_attentions')
S3 = ('It is in this spirit that a majority of American governments have passed new laws since 2009 '
      'making the registration or voting process more difficult.')
x = tok([S3], return_tensors='pt')
pieces = tok.convert_ids_to_tokens(x['input_ids'][0])
with torch.no_grad():
    enc = model.get_encoder()(**x, output_attentions=True, output_hidden_states=True)


@torch.no_grad()
def attention_weights_by_hand(layer, hidden, h=8):
    """softmax(Q K^T / sqrt(d_k)) for every head of one self-attention layer. hidden: (1, n, d_model)."""
    n, d_model = hidden.shape[1], hidden.shape[2]
    d_k = d_model // h
    Q = layer.self_attn.q_proj(hidden)                       # (1, n, d_model)
    K = layer.self_attn.k_proj(hidden)
    Q = Q.view(1, n, h, d_k).transpose(1, 2)                 # (1, h, n, d_k): split into heads
    K = K.view(1, n, h, d_k).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)        # (1, h, n, n): every query against every key
    return torch.softmax(scores, dim=-1)                     # each row sums to 1


L = 4                                                        # layer 5 of 6, counted from 0
ours = attention_weights_by_hand(model.get_encoder().layers[L], enc.hidden_states[L])   # input of layer 5
lib = enc.attentions[L]
log(f'{len(pieces)} tokens; shape {tuple(ours.shape)}; by hand vs output_attentions: max abs difference '
    f'{(ours - lib).abs().max().item():.1e}')
R['attn_diff'] = (ours - lib).abs().max().item()

SHADES = ' .:-=+*#%@'                                       # 10 levels of ink, light to dark


def text_heatmap(rows, labels):
    """One line per head; one character per token; darker = more weight."""
    lines = []
    for name, row in zip(labels, rows):
        lines.append(f'{name:>7s} |' + ''.join(SHADES[min(9, int(w * 10))] for w in row.tolist()) + '|')
    return lines


i = pieces.index('▁making')
for line in text_heatmap(ours[0, :, i], [f'head {k + 1}' for k in range(8)]):
    log(line)
log(f'columns: {len(pieces)} tokens; "making" is column {i + 1}, "more" is {pieces.index("▁more") + 1}, '
    f'"difficult" is {pieces.index("▁difficult") + 1}')
save('part6_code', R)
