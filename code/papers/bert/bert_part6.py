"""Part 6: BERT's limits and how it is used today, measured on real public models.

1. The 512-token limit: the position table has 512 rows; longer inputs must be cut.
2. Footnote 6 in practice: raw BERT [CLS] vectors are poor sentence embeddings; a model fine-tuned
   for similarity (all-MiniLM-L6-v2, a small distilled BERT-architecture model) separates related from unrelated pairs.
3. Two masks at once: BERT predicts every [MASK] separately (the "independence" critique from XLNet).
4. A use case today: named entity recognition with a public BERT-base checkpoint fine-tuned on CoNLL-2003.
Runs on the CPU so the numbers are reproducible."""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, BertForMaskedLM, pipeline
from common import Log, save

torch.manual_seed(0)
log = Log('part6')
out = {}

# ---------------------------------------------------------------- 1. the 512-token limit
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = AutoModel.from_pretrained('bert-base-uncased').eval()
pos_rows = tuple(bert.embeddings.position_embeddings.weight.shape)
long_text = ' '.join(['BERT reads the whole input at once, so every token can look at every other token.'] * 40)
n_full = len(tok(long_text).input_ids)
n_cut = len(tok(long_text, truncation=True, max_length=512).input_ids)
try:
    with torch.no_grad():
        bert(**tok(long_text, return_tensors='pt'))
    err = 'no error'
except Exception as e:  # the position table has no row for position 512 and beyond
    err = f'{type(e).__name__}: {str(e).splitlines()[0][:110]}'
log('1. The 512-token limit')
log(f'   position embedding table: {pos_rows[0]} rows x {pos_rows[1]} numbers')
log(f'   a long text has {n_full} tokens; with truncation it keeps {n_cut}; the rest is thrown away')
log(f'   feeding all {n_full} tokens: {err}')
log('')
out['limit'] = dict(position_rows=pos_rows[0], hidden=pos_rows[1], long_tokens=n_full, kept=n_cut, error=err)

# ---------------------------------------------------------------- 2. sentence similarity
PAIRS = [
    ('A man is playing a guitar.', 'A person is making music.', 'related'),
    ('How do I reset my password?', 'I forgot my login details.', 'related'),
    ('The weather is lovely today.', 'It is sunny and warm outside.', 'related'),
    ('A man is playing a guitar.', 'The stock market fell sharply.', 'unrelated'),
    ('How do I reset my password?', 'Penguins live in Antarctica.', 'unrelated'),
    ('The weather is lovely today.', 'He parked the truck in the garage.', 'unrelated'),
]
st_name = 'sentence-transformers/all-MiniLM-L6-v2'
st_cfg = AutoConfig.from_pretrained(st_name)
st_tok = AutoTokenizer.from_pretrained(st_name)
st = AutoModel.from_pretrained(st_name).eval()


def mean_pool(model, tk, s):
    enc = tk(s, return_tensors='pt')
    with torch.no_grad():
        h = model(**enc).last_hidden_state[0]
    m = enc.attention_mask[0].unsqueeze(-1).float()
    return (h * m).sum(0) / m.sum()


def cls_vec(s):
    with torch.no_grad():
        return bert(**tok(s, return_tensors='pt')).last_hidden_state[0, 0]


cos = lambda a, b: float(F.cosine_similarity(a, b, dim=0))
log('2. Sentence similarity (cosine, 1 = same direction)')
log(f'   {st_name}: model_type={st_cfg.model_type}, layers={st_cfg.num_hidden_layers}, hidden={st_cfg.hidden_size}, '
    f'params={sum(p.numel() for p in st.parameters()) / 1e6:.1f}M')
log(f'   {"pair (R = related, U = unrelated)":<67} {"raw [CLS]":>9} {"raw mean":>9} {"MiniLM":>7}')
rows = []
for a, b, kind in PAIRS:
    r = dict(a=a, b=b, kind=kind, cls=round(cos(cls_vec(a), cls_vec(b)), 3),
             mean=round(cos(mean_pool(bert, tok, a), mean_pool(bert, tok, b)), 3),
             st=round(cos(mean_pool(st, st_tok, a), mean_pool(st, st_tok, b)), 3))
    rows.append(r)
    log(f'   {kind[0].upper()} {a:<28} | {b:<34} {r["cls"]:>9.3f} {r["mean"]:>9.3f} {r["st"]:>7.3f}')
for key in ('cls', 'mean', 'st'):
    rel = [r[key] for r in rows if r['kind'] == 'related']
    unr = [r[key] for r in rows if r['kind'] == 'unrelated']
    out.setdefault('gap', {})[key] = dict(related=round(sum(rel) / 3, 3), unrelated=round(sum(unr) / 3, 3),
                                          lowest_related=min(rel), highest_unrelated=max(unr))
g = out['gap']
log(f'   average related / unrelated:  raw [CLS] {g["cls"]["related"]:.3f} / {g["cls"]["unrelated"]:.3f},'
    f'  raw mean {g["mean"]["related"]:.3f} / {g["mean"]["unrelated"]:.3f},  MiniLM {g["st"]["related"]:.3f} / {g["st"]["unrelated"]:.3f}')
log('')
out['similarity'] = rows
out['st_model'] = dict(name=st_name, model_type=st_cfg.model_type, layers=st_cfg.num_hidden_layers, hidden=st_cfg.hidden_size,
                       params=sum(p.numel() for p in st.parameters()))

# ---------------------------------------------------------------- 3. two masks are predicted separately
mlm = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
log('3. Two [MASK]s at once: each one is predicted on its own')
two = []
for s in ['i flew from [MASK] [MASK] to london last week.']:
    enc = tok(s, return_tensors='pt')
    pos = (enc.input_ids[0] == tok.mask_token_id).nonzero().flatten().tolist()
    with torch.no_grad():
        p = torch.softmax(mlm(**enc).logits[0], -1)
    tops = []
    for q in pos:
        v, i = p[q].topk(5)
        tops.append([(tok.decode([int(t)]), round(float(x), 3)) for x, t in zip(v, i)])
    log(f'   {s}')
    for k, t in enumerate(tops):
        log(f'     mask {k + 1}: ' + ', '.join(f'{w} {x:.3f}' for w, x in t))
    two.append(dict(sentence=s, top5=tops))
# now fill the first blank ourselves and ask again: the second blank changes completely
cond = []
for first in ['new', 'los', 'hong']:
    s = f'i flew from {first} [MASK] to london last week.'
    enc = tok(s, return_tensors='pt')
    q = (enc.input_ids[0] == tok.mask_token_id).nonzero().item()
    with torch.no_grad():
        p = torch.softmax(mlm(**enc).logits[0, q], -1)
    v, i = p.topk(3)
    t = [(tok.decode([int(k)]), round(float(x), 3)) for x, k in zip(v, i)]
    cond.append(dict(first=first, top3=t))
    log(f'   {s:<44} -> ' + ', '.join(f'{w} {x:.3f}' for w, x in t))
log('')
out['conditional'] = cond
out['two_masks'] = two

# ---------------------------------------------------------------- 4. NER, a use case today
ner_name = 'dslim/bert-base-NER'
ner = pipeline('ner', model=ner_name, aggregation_strategy='simple', device='cpu')
text = 'Ada Lovelace was born in London and worked with Charles Babbage on the Analytical Engine.'
ents = [dict(word=e['word'], group=e['entity_group'], score=round(float(e['score']), 3)) for e in ner(text)]
log(f'4. Named entities with {ner_name} (BERT-base-cased fine-tuned on CoNLL-2003)')
log(f'   {text}')
for e in ents:
    log(f'     {e["word"]:<22} {e["group"]:<5} {e["score"]:.3f}')
raw = pipeline('ner', model=ner_name, aggregation_strategy='none', device='cpu')(text)
tail = [dict(token=e['word'], label=e['entity'], score=round(float(e['score']), 3)) for e in raw if e['start'] >= text.index('Analytical')]
log('   token by token, for "Analytical Engine":')
for e in tail:
    log(f'     {e["token"]:<22} {e["label"]:<6} {e["score"]:.3f}')
out['ner'] = dict(model=ner_name, text=text, entities=ents, analytical_engine_tokens=tail)

save('part6', out)
