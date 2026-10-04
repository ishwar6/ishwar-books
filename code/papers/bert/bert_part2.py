"""Part 2: inside BERT. Counts the parameters by formula and by the real model, takes the vocabulary apart,
runs the real WordPiece tokenizer, rebuilds the input embeddings by hand, and shows that the same word gets
different vectors in different sentences. Runs on the CPU."""
import re, torch
import torch.nn.functional as Fn
from transformers import BertConfig, BertModel, BertTokenizer
from common import Log, save

torch.manual_seed(0)
log = Log('part2')
R = {}

tok = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').eval()


# ---------------------------------------------------------------- 1. parameter count by formula
def formula(V, P, S, H, L, I):
    emb = V * H + P * H + S * H + 2 * H               # token, position, segment tables + LayerNorm (scale, shift)
    attn = 4 * (H * H + H)                              # Q, K, V and output projections, each with a bias
    ffn = H * I + I + I * H + H                         # H -> 4H -> H, with biases
    norms = 2 * 2 * H                                   # two LayerNorms per layer
    layer = attn + ffn + norms
    pooler = H * H + H
    return dict(embeddings=emb, attention_per_layer=attn, ffn_per_layer=ffn, layernorm_per_layer=norms,
                per_layer=layer, all_layers=L * layer, pooler=pooler, total=emb + L * layer + pooler)


def count(m):
    return sum(p.numel() for p in m.parameters())


log('== 1. parameters: formula vs the real model ==')
for name in ['bert-base-uncased', 'bert-large-uncased']:
    c = BertConfig.from_pretrained(name)
    f = formula(c.vocab_size, c.max_position_embeddings, c.type_vocab_size, c.hidden_size, c.num_hidden_layers, c.intermediate_size)
    if name == 'bert-base-uncased':
        real = model
    else:
        with torch.device('meta'):                      # build the shapes only, no weights downloaded or stored
            real = BertModel(c)
    n = count(real)
    parts = dict(embeddings=count(real.embeddings), encoder=count(real.encoder), pooler=count(real.pooler))
    R[name] = dict(config=dict(L=c.num_hidden_layers, H=c.hidden_size, A=c.num_attention_heads, ffn=c.intermediate_size,
                               vocab=c.vocab_size, positions=c.max_position_embeddings, segments=c.type_vocab_size),
                   formula=f, counted=n, counted_parts=parts, match=(n == f['total']))
    log(f'{name}: L={c.num_hidden_layers} H={c.hidden_size} A={c.num_attention_heads} feed-forward={c.intermediate_size}')
    log(f'  embeddings {f["embeddings"]:>12,}   one layer {f["per_layer"]:>11,}   x{c.num_hidden_layers} = {f["all_layers"]:>12,}   pooler {f["pooler"]:>9,}')
    log(f'  formula total {f["total"]:>12,}   counted in the model {n:>12,}   same: {n == f["total"]}')
    log(f'  without the pooler {f["total"] - f["pooler"]:,}')
ffn_share = R['bert-base-uncased']['formula']
log(f'  in one base layer: attention {ffn_share["attention_per_layer"]:,}, feed-forward {ffn_share["ffn_per_layer"]:,}, layer norms {ffn_share["layernorm_per_layer"]:,}')
log('')

# ---------------------------------------------------------------- 2. the vocabulary
log('== 2. the WordPiece vocabulary of bert-base-uncased ==')
vocab = sorted(tok.vocab, key=tok.vocab.get)          # in id order, as in vocab.txt
groups = {
    'special ([PAD] [UNK] [CLS] [SEP] [MASK])': [t for t in vocab if t in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')],
    'unused placeholders ([unused0] ...)': [t for t in vocab if re.fullmatch(r'\[unused\d+\]', t)],
    'single characters': [t for t in vocab if len(t) == 1],
    '## pieces of 1 character (##a ...)': [t for t in vocab if t.startswith('##') and len(t) == 3],
    '## pieces, longer (##ing ...)': [t for t in vocab if t.startswith('##') and len(t) > 3],
    'whole tokens, 2+ characters': [t for t in vocab if not t.startswith('##') and not t.startswith('[') and len(t) > 1],
}
covered = sum(len(v) for v in groups.values())
R['vocab'] = dict(size=len(vocab), groups={k: len(v) for k, v in groups.items()}, covered=covered,
                  ids={t: tok.vocab[t] for t in ('[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]')},
                  examples={k: v[:6] for k, v in groups.items()})
log(f'entries in vocab.txt: {len(vocab):,}')
for k, v in groups.items():
    log(f'  {k:<42} {len(v):>6,}   e.g. {" ".join(v[:5])}')
log(f'  (groups add up to {covered:,})')
log('  ids: ' + ', '.join(f'{t}={i}' for t, i in R['vocab']['ids'].items()))
log('')

# ---------------------------------------------------------------- 3. WordPiece in action
log('== 3. WordPiece splits ==')
words = ['playing', 'likes', 'cute', 'embeddings', 'transformers', 'unaffable', 'tokenization', 'hairy', 'bidirectional']
R['splits'] = {w: tok.tokenize(w) for w in words}
for w, s in R['splits'].items():
    log(f'  {w:<15} -> {" ".join(s)}')
enc = tok('my dog is cute', 'he likes playing')
toks = tok.convert_ids_to_tokens(enc['input_ids'])
R['figure2'] = dict(tokens=toks, input_ids=enc['input_ids'], token_type_ids=enc['token_type_ids'],
                    position_ids=list(range(len(toks))))


def wordpiece(word, vocab, unk='[UNK]'):
    """Greedy longest-match-first, as in the original tokenization.py: take the longest piece that is in the
    vocabulary, then continue with '##' + the longest piece of the rest, until the word is used up."""
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        while end > start:
            piece = ('##' if start > 0 else '') + word[start:end]
            if piece in vocab:
                break
            end -= 1
        if end == start:                      # not even one character matched
            return [unk]
        pieces.append(piece)
        start = end
    return pieces


import pymupdf
page_text = ' '.join(pymupdf.open(__import__('os').path.expanduser('~/.cache/papers/1810.04805.pdf'))[i].get_text() for i in range(9))
# the paper's text contains the literal strings "[CLS]", "[SEP]", "[MASK]". The real tokenizer keeps those as
# special tokens (a step before WordPiece), so we drop the brackets to compare only the WordPiece step.
page_text = re.sub(r'\[(CLS|SEP|MASK|UNK|PAD)\]', r'\1', page_text)
bt = tok.backend_tokenizer                              # step 1+2 of the real tokenizer: lowercase, strip accents, split on spaces and punctuation
words_ = [w for w, _ in bt.pre_tokenizer.pre_tokenize_str(bt.normalizer.normalize_str(page_text))]
ours = [p for w in words_ for p in wordpiece(w, tok.vocab)]
real = tok.tokenize(page_text)
split_words = sorted({w for w in words_ if len(wordpiece(w, tok.vocab)) > 1})
R['wordpiece_check'] = dict(words=len(words_), pieces_ours=len(ours), pieces_real=len(real), identical=(ours == real),
                            words_split=len(split_words), examples=[(w, wordpiece(w, tok.vocab)) for w in split_words[:0]])
log(f'our WordPiece function vs the real tokenizer, pages 1-9 of the BERT paper:')
log(f'  {len(words_):,} words -> {len(ours):,} pieces (ours), {len(real):,} pieces (real); identical: {ours == real}')
log(f'  distinct words that needed more than one piece: {len(split_words):,}')
log('Figure 2 pair ("my dog is cute", "he likes playing"):')
log('  tokens:   ' + ' '.join(f'{t:>7}' for t in toks))
log('  ids:      ' + ' '.join(f'{i:>7}' for i in enc['input_ids']))
log('  segment:  ' + ' '.join(f'{("A" if s == 0 else "B"):>7}' for s in enc['token_type_ids']))
log('  position: ' + ' '.join(f'{i:>7}' for i in range(len(toks))))
log('')

# ---------------------------------------------------------------- 4. the input embedding, rebuilt by hand
log('== 4. input embedding = LayerNorm(token + segment + position) ==')
ids = torch.tensor([enc['input_ids']])
seg = torch.tensor([enc['token_type_ids']])
pos = torch.arange(ids.shape[1]).unsqueeze(0)
E = model.embeddings
with torch.no_grad():
    s = E.word_embeddings(ids) + E.token_type_embeddings(seg) + E.position_embeddings(pos)
    mine = E.LayerNorm(s)
    theirs = E(input_ids=ids, token_type_ids=seg)
    plain_sum_diff = (s - theirs).abs().max().item()
diff = (mine - theirs).abs().max().item()
R['embedding_check'] = dict(max_abs_diff_with_layernorm=diff, max_abs_diff_plain_sum=plain_sum_diff,
                            shape=list(theirs.shape), table_shapes=dict(token=list(E.word_embeddings.weight.shape),
                            segment=list(E.token_type_embeddings.weight.shape), position=list(E.position_embeddings.weight.shape)))
log(f'  tables: token {tuple(E.word_embeddings.weight.shape)}, segment {tuple(E.token_type_embeddings.weight.shape)}, position {tuple(E.position_embeddings.weight.shape)}')
log(f'  output shape {tuple(theirs.shape)} (1 sequence, {ids.shape[1]} tokens, 768 numbers each)')
log(f'  max |difference|, our LayerNorm(sum) vs the model: {diff:.2e}')
log(f'  max |difference|, plain sum without LayerNorm vs the model: {plain_sum_diff:.2f}')
log('')

# ---------------------------------------------------------------- 5. one word, many vectors
log('== 5. "bank": one static vector, many contextual vectors ==')
sents = ['he sat on the river bank and watched the water.',
         'the bank of the river was covered in mud.',
         'she opened a savings account at the bank.',
         'the bank approved my loan yesterday.']
vecs = []
with torch.no_grad():
    for t in sents:
        e = tok(t, return_tensors='pt')
        p = e.input_ids[0].tolist().index(tok.vocab['bank'])
        vecs.append(model(**e).last_hidden_state[0, p])
static = E.word_embeddings.weight[tok.vocab['bank']]
cos = [[round(Fn.cosine_similarity(a, b, dim=0).item(), 3) for b in vecs] for a in vecs]
R['bank'] = dict(sentences=sents, cosine=cos)
log('  the input (static) vector of "bank" is the same row of the table in every sentence: cosine 1.000')
log('  cosine similarity of BERT\'s output vectors for "bank":')
for i, t in enumerate(sents):
    log(f'  [{i + 1}] ' + '  '.join(f'{c:.3f}' for c in cos[i]) + f'   {t}')
same = (cos[0][1] + cos[2][3]) / 2
diff_ = (cos[0][2] + cos[0][3] + cos[1][2] + cos[1][3]) / 4
R['bank'].update(avg_same_meaning=round(same, 3), avg_different_meaning=round(diff_, 3))
log(f'  average, same meaning (river-river, money-money): {same:.3f}')
log(f'  average, different meaning (river-money):         {diff_:.3f}')

log('')

# ---------------------------------------------------------------- 6. C and the pooler
log('== 6. the [CLS] vector C and the released pooler ==')
with torch.no_grad():
    out = model(**tok('my dog is cute', 'he likes playing', return_tensors='pt'))
    C = out.last_hidden_state[0, 0]
    pooled = torch.tanh(model.pooler.dense(C))
    pdiff = (pooled - out.pooler_output[0]).abs().max().item()
R['pooler'] = dict(C_shape=list(C.shape), max_abs_diff=pdiff, pooler_weight_shape=list(model.pooler.dense.weight.shape))
log(f'  C = final hidden vector of [CLS], shape {tuple(C.shape)}')
log(f'  pooler = tanh(W C + b), W shape {tuple(model.pooler.dense.weight.shape)}; max |difference| vs model.pooler_output: {pdiff:.2e}')

log('')

# ---------------------------------------------------------------- 7. one encoder layer, by hand
log('== 7. layer 1 of BERT, recomputed by hand ==')
lay = model.encoder.layer[0]
with torch.no_grad():
    x = E(input_ids=ids, token_type_ids=seg)[0]                          # (10, 768): the input embeddings
    Hh, A = 768, 12
    d = Hh // A                                                            # 64 numbers per head
    sa = lay.attention.self
    q = sa.query(x).view(-1, A, d).transpose(0, 1)                         # (12 heads, 10 tokens, 64)
    k = sa.key(x).view(-1, A, d).transpose(0, 1)
    v = sa.value(x).view(-1, A, d).transpose(0, 1)
    att = torch.softmax(q @ k.transpose(1, 2) / d ** 0.5, dim=-1)         # no mask: every token sees every token
    heads = (att @ v).transpose(0, 1).reshape(-1, Hh)                      # join the 12 heads again
    h1 = lay.attention.output.LayerNorm(x + lay.attention.output.dense(heads))      # add and normalise
    ff = lay.output.dense(Fn.gelu(lay.intermediate.dense(h1)))             # 768 -> 3072 -> 768 with GELU
    mine = lay.output.LayerNorm(h1 + ff)                                   # add and normalise again
    theirs = lay(x.unsqueeze(0))
    theirs = theirs[0] if isinstance(theirs, tuple) else theirs
    ldiff = (mine - theirs[0]).abs().max().item()
R['layer_check'] = dict(max_abs_diff=ldiff, head_size=d, attention_rows_sum_to_one=bool(torch.allclose(att.sum(-1), torch.ones(A, x.shape[0]))))
log(f'  12 heads of 64 numbers each; every attention row sums to 1: {R["layer_check"]["attention_rows_sum_to_one"]}')
log(f'  max |difference|, our layer vs the model\'s layer 1: {ldiff:.2e}')

save('part2', R)
