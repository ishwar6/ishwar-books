"""Background page of BERT, explained: before and after BERT, run on a laptop.

What it shows (every number is printed and saved to results/partb_stdout.txt and results/partb.json):
  1. Counting (n-gram models): a bigram model of WikiText-2 with add-one smoothing. Its perplexity on the test
     text, against a unigram model, and how many test word pairs it never saw in training.
  2. Words as vectors (word2vec): skip-gram with negative sampling, trained from scratch on WikiText-2. The loss
     written by hand (Method 1) is checked against PyTorch's binary cross-entropy (Method 2). Nearest neighbours.
  3. Reading in order (RNN): the gradient that reaches an early word fades with distance. Computed by hand as a
     product of Jacobians (Method 1) and by autograd (Method 2).
  4. Static against contextual vectors: "bank" in our word2vec and in BERT's input table has one vector;
     in BERT's last layer, the river bank and the money bank get different vectors.
  5. After BERT: the scaling law of Kaplan et al. (2020), L(N) = (N_c / N)^alpha_N, at the sizes the GPT-3 paper lists.
  6. The progress curves: GLUE and SuperGLUE scores over time, every number with the paper it comes from.
Runs: word2vec on Apple MPS (if present), everything else on the CPU. Run: python bert_partb.py   (about 3 minutes)"""
import collections, math, random, re
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from common import Log, save

log = Log('partb')
torch.manual_seed(0)
random.seed(0)
GPU = 'mps' if torch.backends.mps.is_available() else 'cpu'
OUT = {}

wt = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')


def words(split):
    """Lower-case words of one split, punctuation dropped."""
    text = ' '.join(wt[split]['text']).lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


train, test = words('train'), words('test')

# ---------------------------------------------------------------- 1. counting: the bigram model
log('== 1. A bigram model of WikiText-2: P(w_t | w_{t-1}) = count(w_{t-1}, w_t) / count(w_{t-1})')
uni = collections.Counter(train)
bi = collections.Counter(zip(train, train[1:]))
V = len(uni) + 1                                           # +1 for words never seen in training
log(f'training text: {len(train):,} words, {len(uni):,} different words, {len(bi):,} different word pairs')


def p_bigram(prev, w):
    """Add-one smoothing: every pair gets one extra imaginary count, so no probability is zero."""
    return (bi[(prev, w)] + 1) / (uni[prev] + V)


def p_unigram(w):
    return (uni[w] + 1) / (len(train) + V)


def p_interp(prev, w, lam=0.7):
    """Interpolation: trust the pair count with weight lam, fall back on the single-word count otherwise."""
    ml = bi[(prev, w)] / uni[prev] if uni[prev] else 0.0
    return lam * ml + (1 - lam) * p_unigram(w)


def perplexity(logp):
    """exp of the average negative log probability: the number of equally likely choices the model is torn between."""
    return math.exp(-sum(logp) / len(logp))


pairs = list(zip(test, test[1:]))
ppl_bi = perplexity([math.log(p_bigram(a, b)) for a, b in pairs])
ppl_uni = perplexity([math.log(p_unigram(b)) for _, b in pairs])
ppl_int = perplexity([math.log(p_interp(a, b)) for a, b in pairs])
unseen = sum((a, b) not in bi for a, b in pairs) / len(pairs)
row = sum(p_bigram('the', w) for w in list(uni) + ['<unk>'])
log(f'test text: {len(test):,} words')
log(f'perplexity, unigram (no context): {ppl_uni:,.0f}')
log(f'perplexity, bigram with add-one smoothing: {ppl_bi:,.0f}')
log(f'perplexity, bigram mixed with unigram (0.7 / 0.3): {ppl_int:,.0f}')
log(f'test word pairs never seen in training: {100 * unseen:.1f}%')
log(f'check: the probabilities after "the" add up to {row:.6f}')
top = sorted(((bi[('the', w)], w) for w in uni if bi[('the', w)]), reverse=True)[:5]
log('most likely words after "the": ' + ', '.join(f'{w} ({p_bigram("the", w):.3f})' for _, w in top))
OUT['bigram'] = {'words': len(train), 'vocab': len(uni), 'ppl_uni': ppl_uni, 'ppl_bi': ppl_bi, 'ppl_int': ppl_int, 'unseen': unseen}

# ---------------------------------------------------------------- 2. word2vec: skip-gram with negative sampling
log('')
log('== 2. word2vec (skip-gram, negative sampling) trained from scratch on WikiText-2')
vocab = [w for w, c in uni.most_common() if c >= 5]
idx = {w: i for i, w in enumerate(vocab)}
freq = torch.tensor([uni[w] for w in vocab], dtype=torch.float)
SUB = 1e-4                                                 # the paper uses about 1e-5 on billions of words
keep = (SUB / (freq / len(train))).sqrt().clamp(max=1)     # keep a word with probability sqrt(t / f(w))
ids = torch.tensor([idx[w] for w in train if w in idx and random.random() < keep[idx[w]]])
noise = freq ** 0.75 / (freq ** 0.75).sum()                # the paper's unigram^(3/4) noise distribution
DIM, WIN, NEG, EPOCHS, BATCH = 100, 2, 5, 5, 4096
log(f'{len(ids):,} training words after subsampling (t = {SUB:g}), vocabulary of {len(vocab):,} words (seen at least 5 times); '
    f'{DIM} dimensions, window {WIN}, {NEG} negatives, {EPOCHS} epochs')

centre, context = [], []
for off in [o for o in range(-WIN, WIN + 1) if o]:          # every (centre word, context word) pair in the window
    a = ids[max(0, -off):len(ids) - max(0, off)]
    b = ids[max(0, off):len(ids) - max(0, -off)]
    centre.append(a); context.append(b)
centre, context = torch.cat(centre), torch.cat(context)
log(f'{len(centre):,} (centre, context) pairs per epoch')

v_in = torch.nn.Embedding(len(vocab), DIM).to(GPU)          # v: the vector of a word as the centre
u_out = torch.nn.Embedding(len(vocab), DIM).to(GPU)         # u: the vector of a word as a context
torch.nn.init.uniform_(v_in.weight, -0.5 / DIM, 0.5 / DIM)
torch.nn.init.zeros_(u_out.weight)
opt = torch.optim.Adam(list(v_in.parameters()) + list(u_out.parameters()), lr=2e-3)
noise_dev = noise.to(GPU)


def sgns_loss_by_hand(c, o, k):
    """Method 1, the paper's objective: -log s(u_o . v_c) - sum over k noise words of log s(-u_k . v_c)."""
    vc = v_in(c)                                            # (B, DIM)
    pos = (u_out(o) * vc).sum(-1)                           # (B,)   score of the real context word
    neg = torch.bmm(u_out(k), vc[:, :, None])[..., 0]       # (B, K) scores of the noise words
    return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(-1)).mean()


def sgns_loss_library(c, o, k):
    """Method 2: the same as binary classification, real pair = 1, noise pair = 0, with PyTorch's BCE."""
    vc = v_in(c)
    logits = torch.cat([(u_out(o) * vc).sum(-1, keepdim=True), torch.bmm(u_out(k), vc[:, :, None])[..., 0]], 1)
    labels = torch.cat([torch.ones_like(logits[:, :1]), torch.zeros_like(logits[:, 1:])], 1)
    return F.binary_cross_entropy_with_logits(logits, labels, reduction='none').sum(-1).mean()


for ep in range(EPOCHS):
    perm = torch.randperm(len(centre))
    total = 0.0
    for s in range(0, len(perm), BATCH):
        j = perm[s:s + BATCH]
        c, o = centre[j].to(GPU), context[j].to(GPU)
        k = torch.multinomial(noise_dev, len(j) * NEG, replacement=True).view(len(j), NEG)
        loss = sgns_loss_by_hand(c, o, k)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(j)
    if ep == 0:
        with torch.no_grad():
            l1, l2 = sgns_loss_by_hand(c, o, k).item(), sgns_loss_library(c, o, k).item()
        log(f'check on one batch: loss by hand {l1:.6f}, library {l2:.6f}')
    log(f'epoch {ep + 1}: mean loss {total / len(perm):.4f}')

E = F.normalize(v_in.weight.detach().cpu(), dim=1)


def neighbours(w, n=6):
    sims = E @ E[idx[w]]
    return [(vocab[i], sims[i].item()) for i in sims.topk(n + 1).indices[1:]]


OUT['w2v'] = {}
for w in ('king', 'france', 'music', 'river', 'bank'):
    nb = neighbours(w)
    OUT['w2v'][w] = nb
    log(f'nearest to {w!r}: ' + ', '.join(f'{x} {s:.2f}' for x, s in nb))
q = E[idx['king']] - E[idx['man']] + E[idx['woman']]
order = (E @ F.normalize(q, dim=0)).argsort(descending=True).tolist()
order = [i for i in order if vocab[i] not in ('king', 'man', 'woman')]
rank = order.index(idx['queen']) + 1
log(f'king - man + woman: closest words {", ".join(vocab[i] for i in order[:5])}; "queen" is number {rank}')
OUT['w2v_analogy'] = {'top': [vocab[i] for i in order[:5]], 'queen_rank': rank}
bank_w2v = E[idx['bank']]

# ---------------------------------------------------------------- 3. RNN: the gradient fades with distance
log('')
log('== 3. A plain RNN, h_t = tanh(W h_{t-1} + U x_t): how much does the last state feel each earlier input?')
torch.manual_seed(0)
H, T = 32, 60
torch.set_default_dtype(torch.float64)                     # 64-bit, so tiny gradients do not round to 0
W = torch.randn(H, H)
W = 0.9 * W / torch.linalg.matrix_norm(W, 2)               # largest stretch of W is 0.9
Uin = torch.randn(H, H) / math.sqrt(H)
xs = torch.randn(T, H, requires_grad=True)
h, hs = torch.zeros(H), []
for t in range(T):
    h = torch.tanh(W @ h + Uin @ xs[t])
    hs.append(h)
target = torch.randn(H)
loss = (hs[-1] * target).sum()

# Method 1: walk back through the steps. d loss / d h_t = J_{t+1}^T ... J_T^T target, with J_t = diag(1 - h_t^2) W
g, by_hand = target.clone(), [None] * T
for t in range(T - 1, -1, -1):
    pre = (1 - hs[t] ** 2) * g                              # through the tanh of step t
    by_hand[t] = (Uin.T @ pre).norm().item()               # gradient that reaches input x_t
    g = W.T @ pre                                          # pass on to h_{t-1}
# Method 2: autograd
loss.backward()
auto = xs.grad.norm(dim=1).tolist()
log(f'hidden size {H}, {T} steps, largest stretch of W = 0.9')
log(f'{"distance from the end":>22} {"gradient size (by hand)":>24} {"autograd":>10}')
OUT['rnn'] = {}
for d in (0, 1, 2, 5, 10, 20, 40, 59):
    t = T - 1 - d
    OUT['rnn'][d] = by_hand[t]
    log(f'{d:>22} {by_hand[t]:>24.2e} {auto[t]:>10.2e}')
log(f'largest relative difference between the two methods: {max(abs(a - b) / b for a, b in zip(by_hand, auto)):.1e}')
OUT['rnn_all'] = by_hand
torch.set_default_dtype(torch.float32)

# ---------------------------------------------------------------- 4. static against contextual vectors
log('')
log('== 4. One vector per word (word2vec, BERT input table) against one vector per use (BERT last layer)')
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased').eval()
S = ['he sat on the bank of the river.', 'she put her money in the bank.', 'the bank approved my loan.',
     'we walked along the bank of the stream.']


@torch.no_grad()
def bank_vectors(s):
    enc = tok(s, return_tensors='pt')
    pos = enc.input_ids[0].tolist().index(tok.convert_tokens_to_ids('bank'))
    out = bert(**enc, output_hidden_states=True)
    table = bert.embeddings.word_embeddings.weight[enc.input_ids[0, pos]]
    return table, out.last_hidden_state[0, pos]


vec = [bank_vectors(s) for s in S]
c = lambda a, b: F.cosine_similarity(a, b, dim=0).item()
log(f'word2vec: "bank" has one vector for every sentence; its neighbours are {", ".join(w for w, _ in OUT["w2v"]["bank"][:4])}')
log(f'BERT input table: cosine of "bank" in sentence 1 and sentence 2: {c(vec[0][0], vec[1][0]):.3f}')
log('BERT last layer, cosine similarity of "bank" between sentences:')
names = ['river', 'money', 'loan', 'stream']
OUT['bank'] = {}
for i in range(4):
    for j in range(i + 1, 4):
        OUT['bank'][f'{names[i]}-{names[j]}'] = c(vec[i][1], vec[j][1])
        log(f'  {S[i]!r:<44} vs {S[j]!r:<44} {c(vec[i][1], vec[j][1]):.3f}')

# ---------------------------------------------------------------- 5. the scaling law
log('')
log('== 5. Kaplan et al. (2020): L(N) = (N_c / N)^alpha_N, alpha_N = 0.076, N_c = 8.8e13 (test loss in nats per token)')
AN, NC = 0.076, 8.8e13
OUT['scaling'] = {}
for name, n in (('GPT (2018)', 1e8), ('BERT-large (2018)', 3e8), ('GPT-2 (2019)', 1.5e9), ('T5 (2019)', 1.1e10),
                ('GPT-3 (2020)', 1.75e11)):
    L = (NC / n) ** AN
    OUT['scaling'][name] = {'n': n, 'loss': L}
    log(f'{name:<18} {n / 1e9:>8.1f}B parameters -> predicted loss {L:.2f}')
log(f'ten times the parameters multiplies the loss by 10^-0.076 = {10 ** -AN:.3f}')
log('(the law is fitted on non-embedding parameters of models trained on WebText2; the sizes above are total counts)')

# ---------------------------------------------------------------- 6. the progress curves (numbers from the papers)
GLUE = [  # (year, month, system, score, where the number is written)
    (2018, 4, 'BiLSTM + ELMo + Attn', 70.0, 'GLUE paper, Table 4 (multi-task)'),
    (2018, 6, 'OpenAI GPT', 72.8, 'BERT paper, §4.1'),
    (2018, 10, 'BERT-large', 80.5, 'BERT paper, §4.1'),
    (2019, 6, 'XLNet', 88.4, 'ALBERT paper, Table 9'),
    (2019, 7, 'RoBERTa', 88.5, 'ALBERT paper, Table 9'),
    (2019, 9, 'ALBERT', 89.4, 'ALBERT paper, Table 9'),
    (2019, 10, 'T5-11B', 90.3, 'T5 paper, Table 14'),
]
SUPERGLUE = [
    (2019, 5, 'BERT-large', 69.0, 'SuperGLUE paper, Table 3'),
    (2019, 10, 'T5-11B', 89.3, 'DeBERTa paper, §1'),
    (2021, 1, 'DeBERTa-1.5B', 89.9, 'DeBERTa paper, §1 (leaderboard, January 2021)'),
]
OUT['glue'], OUT['superglue'] = GLUE, SUPERGLUE
OUT['human'] = {'glue': 87.1, 'superglue': 89.8}
log('')
log('== 6. The progress curves (scores copied from the papers named)')
for name, rows, hum in (('GLUE', GLUE, 87.1), ('SuperGLUE', SUPERGLUE, 89.8)):
    log(f'{name} (human baseline {hum}):')
    for y, m, s, v, src in rows:
        log(f'  {y}-{m:02d}  {s:<22} {v:>5.1f}   {src}')

save('partb', OUT)
log('')
log('saved results/partb.json')
