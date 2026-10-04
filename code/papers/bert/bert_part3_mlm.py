"""Part 3, Task 1: the masked language model.

1. The masking procedure of Section 3.1 / Appendix A.1, written out, and measured over real text.
2. The same procedure as the released create_pretraining_data.py does it (a cap per sequence, never [CLS]/[SEP]).
3. "my dog is hairy" in the three cases of Appendix A.1, with the real bert-base-uncased.
4. Top-1 accuracy of the real model on 15% masked positions of real Wikipedia text.
5. The pre-training loss on a real batch: mean masked-LM loss + mean next-sentence loss.
Runs on the CPU with fixed seeds."""
import random
import torch
import torch.nn.functional as Fn
from transformers import BertForPreTraining, BertTokenizer
from bert_part3_data import articles, sentences
from common import Log, save

log = Log('part3_mlm')
tok = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased').eval()
V = tok.vocab_size
SPECIAL = {tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
CASES = ('mask', 'random', 'same')


def mask_tokens(ids, rng, rate=0.15):
    """The paper's procedure. ids: one sequence of WordPiece ids, [CLS] ... [SEP].
    Choose 15% of the token positions at random. For each chosen position:
    80% -> [MASK], 10% -> a random token, 10% -> unchanged. Returns new ids, chosen positions, their case."""
    cand = [i for i, t in enumerate(ids) if t not in SPECIAL]          # never [CLS], [SEP] or padding
    n = max(1, round(len(cand) * rate))
    chosen = sorted(rng.sample(cand, n))
    out, case = list(ids), {}
    for i in chosen:
        r = rng.random()
        if r < 0.8:
            out[i], case[i] = tok.mask_token_id, 'mask'
        elif r < 0.9:
            out[i], case[i] = rng.randrange(V), 'random'                 # any id in the vocabulary, like the released code
        else:
            case[i] = 'same'
    return out, chosen, case


def released_mask(tokens, rng, rate=0.15, max_pred=20):
    """create_masked_lm_predictions() from google-research/bert, without whole-word masking (the original setting)."""
    cand = [[i] for i, t in enumerate(tokens) if t not in ('[CLS]', '[SEP]')]
    rng.shuffle(cand)
    out = list(tokens)
    n = min(max_pred, max(1, int(round(len(tokens) * rate))))
    chosen = []
    for (i,) in cand[:n]:
        if rng.random() < 0.8:
            out[i] = '[MASK]'
        elif rng.random() < 0.5:
            out[i] = tokens[i]
        else:
            out[i] = tok.convert_ids_to_tokens(rng.randint(0, V - 1))
        chosen.append(i)
    return out, sorted(chosen)


# ---------------------------------------------------------------- real text, cut into 128-token sequences
arts = articles('test')
stream = tok(' '.join(p for a in arts for p in a), add_special_tokens=False).input_ids
L = 128
seqs = [[tok.cls_token_id] + stream[i:i + L - 2] + [tok.sep_token_id] for i in range(0, len(stream) - (L - 2), L - 2)]
log(f'text: WikiText-103 test split, {len(arts)} articles, {len(stream):,} WordPiece tokens -> {len(seqs):,} sequences of {L} tokens')

# ---------------------------------------------------------------- 1. measure the procedure
rng = random.Random(0)
counts = dict(tokens=0, chosen=0, mask=0, random=0, same=0, random_hit_same_id=0)
masked_seqs = []
for s in seqs:
    out, chosen, case = mask_tokens(s, rng)
    masked_seqs.append((s, out, chosen, case))
    counts['tokens'] += sum(t not in SPECIAL for t in s)
    counts['chosen'] += len(chosen)
    for i in chosen:
        counts[case[i]] += 1
        counts['random_hit_same_id'] += case[i] == 'random' and out[i] == s[i]
c = counts
stats = dict(
    chosen_share=c['chosen'] / c['tokens'], mask_share=c['mask'] / c['chosen'], random_share=c['random'] / c['chosen'],
    same_share=c['same'] / c['chosen'], random_of_all=c['random'] / c['tokens'], mask_of_all=c['mask'] / c['tokens'],
    same_of_all=c['same'] / c['tokens'])
log('')
log('1. the masking procedure (paper version), seed 0')
log(f'   word-piece tokens (not counting [CLS]/[SEP]): {c["tokens"]:,}')
log(f'   chosen for prediction: {c["chosen"]:,} = {stats["chosen_share"]:.2%} of tokens')
log(f'   of the chosen: [MASK] {c["mask"]:,} ({stats["mask_share"]:.2%}), random {c["random"]:,} ({stats["random_share"]:.2%}), unchanged {c["same"]:,} ({stats["same_share"]:.2%})')
log(f'   random replacements as a share of ALL tokens: {stats["random_of_all"]:.2%}   (paper: 10% of 15% = 1.5%)')
log(f'   random draws that happened to pick the original token: {c["random_hit_same_id"]}')

# ---------------------------------------------------------------- 2. the released code's version
rng = random.Random(0)
rc = dict(tokens=0, chosen=0, full=0)
for s in seqs:
    toks = tok.convert_ids_to_tokens(s)
    _, chosen = released_mask(toks, rng)
    rc['tokens'] += len(toks) - 2
    rc['chosen'] += len(chosen)
log('')
log('2. the released create_pretraining_data.py logic (max_predictions_per_seq=20, length 128)')
log(f'   predictions per 128-token sequence: round(0.15 x 128) = {round(0.15 * 128)} (cap 20)')
log(f'   chosen: {rc["chosen"]:,} of {rc["tokens"]:,} word-piece tokens = {rc["chosen"] / rc["tokens"]:.2%}')
released = dict(per_seq=round(0.15 * 128), chosen_share=rc['chosen'] / rc['tokens'])


# ---------------------------------------------------------------- 3. my dog is hairy
def predict(text_ids, pos):
    with torch.no_grad():
        logits = model(torch.tensor([text_ids])).prediction_logits[0, pos]
    p = torch.softmax(logits, -1)
    top = p.topk(5)
    return p, [(tok.convert_ids_to_tokens(int(i)), round(float(v), 4)) for v, i in zip(top.values, top.indices)]


# The paper writes the sentence without a full stop. Without one, bert-base-uncased spends position 4 guessing
# the missing "." (it was trained on real text, where sentences end with punctuation), so we add the full stop.
base = tok('my dog is hairy.').input_ids          # [CLS] my dog is hairy . [SEP]
pos = 4
assert tok.convert_ids_to_tokens(base[pos]) == 'hairy'
H, A = tok.convert_tokens_to_ids('hairy'), tok.convert_tokens_to_ids('apple')
pr = lambda x: f'{x:.4f}' if x >= 1e-4 else f'{x:.1e}'
hairy = {}
log('')
log('3. "my dog is hairy." with token 4 chosen (Appendix A.1). What the model predicts at position 4:')
for name, new in (('mask', tok.mask_token_id), ('random', A), ('same', base[pos])):
    ids = list(base)
    ids[pos] = new
    p, top = predict(ids, pos)
    hairy[name] = dict(input=' '.join(tok.convert_ids_to_tokens(ids)), p_hairy=float(p[H]), p_apple=float(p[A]), top5=top)
    log(f'   input: {hairy[name]["input"]}')
    log(f'     P(hairy) = {pr(float(p[H]))}   P(apple) = {pr(float(p[A]))}   top 5: ' + ', '.join(f'{w} {v:.3f}' for w, v in top))

# ---------------------------------------------------------------- 4. accuracy on real text
N_SEQ = 400
hit = {k: 0 for k in CASES}
tot = {k: 0 for k in CASES}
for b in range(0, N_SEQ, 25):
    batch = masked_seqs[b:b + 25]
    with torch.no_grad():
        logits = model(torch.tensor([m[1] for m in batch])).prediction_logits
    pred = logits.argmax(-1)
    for j, (orig, out, chosen, case) in enumerate(batch):
        for i in chosen:
            tot[case[i]] += 1
            hit[case[i]] += int(pred[j, i]) == orig[i]
acc = {k: hit[k] / tot[k] for k in CASES}
n_all = sum(tot.values())
acc_all = sum(hit.values()) / n_all
log('')
log(f'4. top-1 accuracy of bert-base-uncased on the chosen positions of {N_SEQ} sequences ({n_all:,} predictions)')
log(f'   all chosen positions: {acc_all:.1%}')
for k, name in zip(CASES, ('[MASK]', 'random token', 'unchanged')):
    log(f'   {name:13s} {acc[k]:.1%}  ({hit[k]:,} of {tot[k]:,})')

# ---------------------------------------------------------------- 5. the pre-training loss on a real batch
rng = random.Random(1)
pairs = []
docs = [sum((sentences(p) for p in a), []) for a in arts]
docs = [d for d in docs if len(d) >= 4]
for k in range(8):
    d = docs[k]
    a = d[1]
    if k % 2 == 0:
        bsent, label = d[2], 0                                       # IsNext (label 0 in the released code)
    else:
        bsent, label = docs[(k + 20) % len(docs)][3], 1              # NotNext: from another article
    pairs.append((a, bsent, label))
enc = tok([p[0] for p in pairs], [p[1] for p in pairs], padding='max_length', truncation=True, max_length=64, return_tensors='pt')
ids = enc.input_ids.clone()
labels = torch.full_like(ids, -100)                                   # -100 = "not a prediction position"
for r in range(len(pairs)):
    real = ids[r][enc.attention_mask[r].bool()].tolist()
    out, chosen, _ = mask_tokens(real, rng)
    for i in chosen:
        labels[r, i] = real[i]
    ids[r, :len(out)] = torch.tensor(out)
nsp_label = torch.tensor([p[2] for p in pairs])
with torch.no_grad():
    o = model(input_ids=ids, token_type_ids=enc.token_type_ids, attention_mask=enc.attention_mask, labels=labels, next_sentence_label=nsp_label)
mlm = Fn.cross_entropy(o.prediction_logits.view(-1, V), labels.view(-1), ignore_index=-100)
nsp = Fn.cross_entropy(o.seq_relationship_logits, nsp_label)
log('')
log(f'5. pre-training loss on one real batch of {len(pairs)} sentence pairs ({int((labels != -100).sum())} masked positions)')
log(f'   mean masked-LM loss       {mlm.item():.4f}')
log(f'   mean next-sentence loss   {nsp.item():.4f}')
log(f'   sum                       {(mlm + nsp).item():.4f}')
log(f'   loss computed by BertForPreTraining: {o.loss.item():.4f}')

save('part3_mlm', dict(text=dict(articles=len(arts), tokens=len(stream), sequences=len(seqs), seq_len=L),
                       counts=counts, stats=stats, released=released, hairy=hairy,
                       accuracy=dict(n_seq=N_SEQ, n=n_all, all=acc_all, by_case=acc, hits=hit, totals=tot),
                       loss=dict(pairs=len(pairs), masked=int((labels != -100).sum()), mlm=mlm.item(), nsp=nsp.item(), total=o.loss.item())))
