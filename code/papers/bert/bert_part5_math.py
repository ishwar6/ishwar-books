"""Part 5: the equations of the ablation section, worked through with real numbers.

1. Perplexity: one sentence, three masked words, the real bert-base-uncased probabilities -> mean NLL -> exp.
   (and GPT-2 on the same words from the left side only, for contrast)
2. Parameter count, term by term, for every row of Table 6, checked against a real BertModel (meta device);
   the same formula applied to the two older Transformers the paper compares with.
3. ELMo-style LTR + RTL concatenation versus one bidirectional model: parameters, vector size and measured time.
4. The feature-based NER tagger: the shape of every tensor and its parameter count, checked against PyTorch.
5. BIO tags and the first sub-token on a real CoNLL-2003 dev sentence.
6. Why a CRF helps: a hand-made 3-token example decoded with and without the "I-PER cannot follow O" rule.
7. Appendix C.1: predictions per batch for the masked LM versus a left-to-right LM.
"""
import math, time, statistics
import torch
from transformers import AutoTokenizer, BertForMaskedLM, GPT2LMHeadModel, BertConfig, BertModel, AutoModel
from common import Log, save

log = Log('part5_math')
out = {}
torch.manual_seed(0)

# ------------------------------------------------------------------ 1. perplexity, worked
log('1. Perplexity on one sentence, three masked words (bert-base-uncased)')
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
mlm = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
gtok = AutoTokenizer.from_pretrained('openai-community/gpt2')
gpt2 = GPT2LMHeadModel.from_pretrained('openai-community/gpt2').eval()
out['ppl'] = []
for sent, targets in [('she opened the door with her key and walked into the kitchen .', ['door', 'key', 'kitchen']),
                      ('the cat sat on the mat because it was tired .', ['cat', 'mat', 'tired'])]:
    words = sent.split()
    masked = ' '.join('[MASK]' if w in targets else w for w in words)
    enc = tok(masked, return_tensors='pt')
    pos = (enc.input_ids[0] == tok.mask_token_id).nonzero().flatten().tolist()
    with torch.no_grad():
        probs = torch.softmax(mlm(**enc).logits[0, pos], -1)
    log(f'  input: {masked}')
    rows = []
    for p_i, w in zip(probs, targets):
        p = p_i[tok.convert_tokens_to_ids(w)].item()
        top = tok.convert_ids_to_tokens(p_i.topk(3).indices.tolist())
        rows.append(dict(word=w, p=round(p, 4), nll=round(-math.log(p), 4), top3=top))
        log(f'  {w:8s} p = {p:.4f}   -log p = {-math.log(p):.4f}   (top 3 guesses: {", ".join(top)})')
    mean_nll = sum(-math.log(r['p']) for r in rows) / len(rows)
    terms_txt = ' + '.join(f"{r['nll']:.4f}" for r in rows)
    log(f'  mean NLL = ({terms_txt}) / 3 = {mean_nll:.4f} nats -> perplexity = exp({mean_nll:.4f}) = {math.exp(mean_nll):.2f}')
    grows = []
    for w in targets:
        left = ' '.join(words[:words.index(w)])
        ids = gtok(left, return_tensors='pt').input_ids
        with torch.no_grad():
            p = torch.softmax(gpt2(ids).logits[0, -1], -1)[gtok.encode(' ' + w)[0]].item()
        grows.append(dict(word=w, left=left, p=round(p, 4), nll=round(-math.log(p), 4)))
        log(f'  GPT-2, left side only: "{left} ___"  p({w}) = {p:.4f}   -log p = {-math.log(p):.4f}')
    g_nll = sum(-math.log(r['p']) for r in grows) / len(grows)
    log(f'  GPT-2 mean NLL = {g_nll:.4f} -> perplexity {math.exp(g_nll):.2f}  (same three words, left context only)')
    out['ppl'].append(dict(sentence=sent, masked=masked, bert=rows, bert_mean_nll=round(mean_nll, 4), bert_ppl=round(math.exp(mean_nll), 2),
                           gpt2=grows, gpt2_mean_nll=round(g_nll, 4), gpt2_ppl=round(math.exp(g_nll), 2)))
    log('')
log(f'  sanity check: a model that is uniform over 4 words has NLL = log 4 = {math.log(4):.4f} and perplexity {math.exp(math.log(4)):.1f}')
log('')

# ------------------------------------------------------------------ 2. parameter count, term by term
log('2. Parameters term by term (V = 30,522 tokens, 512 positions, 2 segments, feed-forward 4H)')
V, P, S = 30522, 512, 2


def terms(L, H):
    emb = V * H + P * H + S * H + 2 * H                     # three tables + LayerNorm (gain and bias)
    attn = 4 * (H * H + H)                                  # Q, K, V and output projections, with biases
    ffn = (H * 4 * H + 4 * H) + (4 * H * H + H)             # H -> 4H -> H, with biases
    ln = 2 * (2 * H)                                        # two LayerNorms per layer
    pool = H * H + H
    return dict(embeddings=emb, attention=L * attn, feed_forward=L * ffn, layer_norms=L * ln, pooler=pool,
                per_layer=attn + ffn + ln, total=emb + L * (attn + ffn + ln) + pool)


out['params'] = []
log(f'{"L":>3s} {"H":>5s} {"A":>3s} {"embeddings":>12s} {"attention":>12s} {"feed-fwd":>12s} {"LayerNorm":>10s} {"pooler":>10s} {"total":>13s} {"counted":>13s}')
for L, H, A in [(3, 768, 12), (6, 768, 3), (6, 768, 12), (12, 768, 12), (12, 1024, 16), (24, 1024, 16)]:
    t = terms(L, H)
    cfg = BertConfig(vocab_size=V, hidden_size=H, num_hidden_layers=L, num_attention_heads=A, intermediate_size=4 * H)
    with torch.device('meta'):
        n = sum(p.numel() for p in BertModel(cfg).parameters())
    assert n == t['total']
    out['params'].append(dict(L=L, H=H, A=A, **t, counted=n))
    log(f'{L:3d} {H:5d} {A:3d} {t["embeddings"]:12,d} {t["attention"]:12,d} {t["feed_forward"]:12,d} {t["layer_norms"]:10,d} {t["pooler"]:10,d} {t["total"]:13,d} {n:13,d}')
log(f'per layer: 12H^2 + 13H = {terms(1, 768)["per_layer"]:,} for H = 768 and {terms(1, 1024)["per_layer"]:,} for H = 1024')
for name, L, H, said in [('Vaswani et al. 2017, big encoder', 6, 1024, '100M for the encoder'), ('Al-Rfou et al. 2018', 64, 512, '235M')]:
    t = terms(L, H)
    log(f'{name}: L={L}, H={H}: the layers alone, if shaped like BERT\'s (12H^2 + 13H each) = {L * t["per_layer"]:,}   (paper says {said})')
    out[f'layers_only_{L}x{H}'] = L * t['per_layer']
log('')

# ------------------------------------------------------------------ 3. ELMo-style concatenation versus one bidirectional model
log('3. Two one-way models glued together (ELMo style) versus one bidirectional model, BERT-base size')
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
enc_model = BertModel.from_pretrained('bert-base-uncased').to(dev).eval()
n_one = sum(p.numel() for p in enc_model.parameters())
x = torch.randint(1000, 30000, (8, 512), device=dev)
causal = torch.tril(torch.ones(512, 512, dtype=torch.bool, device=dev))   # LTR: row i sees columns 0..i
anti = torch.triu(torch.ones(512, 512, dtype=torch.bool, device=dev))     # RTL: row i sees columns i..511


def run(mask=None):
    with torch.no_grad():
        if mask is None:
            h = enc_model(input_ids=x).last_hidden_state
        else:   # a 4D mask (batch, 1, query, key): True = may attend
            h = enc_model(input_ids=x, attention_mask=mask[None, None].expand(8, 1, -1, -1)).last_hidden_state
    if dev == 'mps':
        torch.mps.synchronize()
    return h


run(); run(causal)
t_bi, t_two = [], []
for _ in range(5):
    t0 = time.time(); h_bi = run(); t_bi.append(time.time() - t0)
    t0 = time.time(); h_l = run(causal); h_r = run(anti); t_two.append(time.time() - t0)
h_cat = torch.cat([h_l, h_r], -1)
log(f'  parameters: one bidirectional model {n_one:,}; LTR + RTL = {2 * n_one:,}')
log(f'  vector per token: bidirectional {tuple(h_bi.shape[1:])} ; LTR {tuple(h_l.shape[1:])} + RTL {tuple(h_r.shape[1:])} -> concat {tuple(h_cat.shape[1:])}')
log(f'  time for a batch of 8 x 512 tokens on {dev} (median of 5): one model {statistics.median(t_bi) * 1000:.0f} ms, LTR + RTL {statistics.median(t_two) * 1000:.0f} ms '
    f'-> {statistics.median(t_two) / statistics.median(t_bi):.2f}x')
out['concat_cost'] = dict(device=dev, params_one=n_one, params_two=2 * n_one, ms_one=round(statistics.median(t_bi) * 1000),
                          ms_two=round(statistics.median(t_two) * 1000), ratio=round(statistics.median(t_two) / statistics.median(t_bi), 2))
# what can token i see after the LAST layer, and what can it see INSIDE layer k?
n = 6
log(f'  in a sequence of {n} tokens, which positions can token 3 use (1-based)?')
log(f'    bidirectional, any layer: {list(range(1, n + 1))}')
log(f'    LTR stack, any layer:     {list(range(1, 4))}    RTL stack, any layer: {list(range(3, n + 1))}')
log('    concatenation: both lists, but only side by side at the very top; no layer ever mixes them')
del enc_model
log('')

# ------------------------------------------------------------------ 4. the feature-based tagger, shape by shape
log('4. The feature-based NER tagger (Section 5.3): shapes and parameters')
ctok = AutoTokenizer.from_pretrained('bert-base-cased')
cased = AutoModel.from_pretrained('bert-base-cased').eval()
ws = ['Ada', 'Lovelace', 'met', 'Charles', 'Babbage', 'in', 'London', '.']
e = ctok(ws, is_split_into_words=True, return_tensors='pt')
with torch.no_grad():
    hs = torch.stack(cased(**e, output_hidden_states=True).hidden_states, 1)[0]     # (tokens, 13, 768)
wid = e.word_ids(0)
first = [t for t, w in enumerate(wid) if w is not None and (t == 0 or wid[t - 1] != w)]
feats = hs[first]                                                                   # (words, 13, 768)
cat4 = feats[:, 9:13].flatten(1)                                                    # (words, 3072)
lstm = torch.nn.LSTM(3072, 384, num_layers=2, bidirectional=True, batch_first=True)
clf = torch.nn.Linear(768, 9)
with torch.no_grad():
    hseq, _ = lstm(cat4[None])
    logits = clf(hseq[0])
shapes = [('WordPiece tokens', list(e.input_ids.shape[1:])), ('all hidden states', list(hs.shape)), ('first sub-token of each word', list(feats.shape)),
          ('concat last four layers', list(cat4.shape)), ('BiLSTM output', list(hseq[0].shape)), ('tag scores', list(logits.shape))]
log(f'  sentence: {" ".join(ws)}')
log(f'  WordPiece: {ctok.convert_ids_to_tokens(e.input_ids[0])}')
for k, s in shapes:
    log(f'  {k:30s} {s}')


def lstm_params(inp, h):   # PyTorch keeps two bias vectors per gate set
    return 4 * (h * inp + h * h + 2 * h)


p1, p2 = 2 * lstm_params(3072, 384), 2 * lstm_params(768, 384)
n_lstm = sum(p.numel() for p in lstm.parameters())
n_clf = sum(p.numel() for p in clf.parameters())
assert n_lstm == p1 + p2 and n_clf == 9 * 768 + 9
log(f'  BiLSTM layer 1 (input 3072): 2 directions x 4 x (384*3072 + 384*384 + 2*384) = {p1:,}')
log(f'  BiLSTM layer 2 (input 768):  2 directions x 4 x (384*768 + 384*384 + 2*384)  = {p2:,}')
log(f'  classifier 9 x 768 + 9 = {n_clf:,};  tagger total {n_lstm + n_clf:,} trainable, BERT-base-cased frozen ({sum(p.numel() for p in cased.parameters()):,} parameters never change)')
out['tagger'] = dict(sentence=ws, wordpieces=ctok.convert_ids_to_tokens(e.input_ids[0]), shapes=dict(shapes), lstm_layer1=p1, lstm_layer2=p2,
                     classifier=n_clf, tagger_total=n_lstm + n_clf, bert_frozen=sum(p.numel() for p in cased.parameters()))
log('')

# ------------------------------------------------------------------ 5. BIO tags and the first sub-token on a real dev sentence
log('5. BIO tags and first sub-tokens on a real CoNLL-2003 dev sentence')
from datasets import load_dataset
ds = load_dataset('eriktks/conll2003', revision='refs/convert/parquet')['validation']
names = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
pick = None
for i, ex in enumerate(ds):
    tags = [names[t] for t in ex['ner_tags']]
    if 7 <= len(ex['tokens']) <= 11 and 'I-PER' in tags and ('B-LOC' in tags or 'B-ORG' in tags):
        pieces = [ctok.tokenize(w) for w in ex['tokens']]
        if sum(len(p) > 1 for p in pieces) >= 2:
            pick = (i, ex['tokens'], tags, pieces)
            break
i, toks, tags, pieces = pick
log(f'  dev sentence #{i}: {" ".join(toks)}')
for w, t, p in zip(toks, tags, pieces):
    log(f'  {w:14s} {t:7s} pieces {p}  -> the tagger reads "{p[0]}"')
out['bio'] = dict(index=i, words=toks, tags=tags, pieces=pieces)
log('')

# ------------------------------------------------------------------ 6. CRF: a hand-made 3-token example
log('6. Per-token argmax versus a decoder that knows "I-PER cannot follow O" (hand-made scores, for illustration)')
T = ['O', 'B-PER', 'I-PER']
words6 = ['met', 'Ada', 'Lovelace']
em = torch.tensor([[3.0, 0.5, 0.2],     # met
                   [1.2, 1.0, 0.4],     # Ada: O narrowly beats B-PER
                   [0.3, 0.6, 2.5]])    # Lovelace: clearly a continuation
greedy = [T[k] for k in em.argmax(1).tolist()]
allowed = torch.zeros(3, 3)
allowed[0, 2] = float('-inf')            # O -> I-PER is not allowed
best, best_s = None, -1e9
import itertools
for seq in itertools.product(range(3), repeat=3):
    if seq[0] == 2:
        continue                          # cannot start with I-PER either
    s = sum(em[k, seq[k]].item() for k in range(3)) + sum(allowed[seq[k], seq[k + 1]].item() for k in range(2))
    if s > best_s:
        best, best_s = seq, s
g_score = sum(em[k, em.argmax(1)[k]].item() for k in range(3))
log(f'  scores (rows = words {words6}, columns = {T}): ' + '; '.join(', '.join(f'{v:.1f}' for v in r) for r in em.tolist()))
log(f'  per-token argmax: {greedy}  (sum {g_score:.1f}; O followed by I-PER is invalid)')
log(f'  best valid sequence: {[T[k] for k in best]}  (sum {best_s:.1f})')
out['crf'] = dict(words=words6, tags=T, scores=em.tolist(), greedy=greedy, greedy_sum=round(g_score, 1), best=[T[k] for k in best], best_sum=round(best_s, 1))
log('')

# ------------------------------------------------------------------ 7. Appendix C.1: predictions per batch
log('7. Training signal per batch (Appendix C.1)')
tokens = 256 * 512
mlm_pred = round(0.15 * tokens)
log(f'  one batch = 256 x 512 = {tokens:,} tokens; masked LM predicts 15% = {mlm_pred:,}; left-to-right LM predicts every next token, about {tokens:,}')
log(f'  over 1,000,000 steps: {mlm_pred * 1_000_000:,} masked-LM predictions versus about {tokens * 1_000_000:,} for LTR ({tokens / mlm_pred:.2f}x more)')
out['signal'] = dict(tokens=tokens, mlm=mlm_pred, ratio=round(tokens / mlm_pred, 2))
save('part5_math', out)
