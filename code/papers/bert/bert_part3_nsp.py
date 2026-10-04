"""Part 3, Task 2: next sentence prediction, with the real pre-trained NSP head of bert-base-uncased.

1. The two examples of Appendix A.1 (with their [MASK] tokens), and a few pairs of our own.
2. Accuracy on 1,000 real sentence pairs from Wikipedia: 500 IsNext (the true next sentence)
   and 500 NotNext (a random sentence from a different article).
3. Footnote 6: cosine similarity of the raw [CLS] vectors, for related and unrelated sentences.
Runs on the CPU with a fixed seed."""
import random
import torch
from transformers import BertForPreTraining, BertTokenizer
from bert_part3_data import articles, sentences
from common import Log, save

log = Log('part3_nsp')
tok = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForPreTraining.from_pretrained('bert-base-uncased').eval()


def is_next_prob(a, b):
    """P(IsNext) from the NSP head. In the released model, class 0 = IsNext, class 1 = NotNext."""
    enc = tok(a, b, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**enc).seq_relationship_logits[0]
    return float(torch.softmax(logits, -1)[0])


# ---------------------------------------------------------------- 1. examples
examples = [
    ('paper A.1, labelled IsNext', 'the man went to [MASK] store', 'he bought a gallon [MASK] milk'),
    ('paper A.1, labelled NotNext', 'the man [MASK] to the store', 'penguin [MASK] are flight ##less birds'),
    ('ours, true next', 'she opened the fridge.', 'there was nothing left but an old lemon.'),
    ('ours, random', 'she opened the fridge.', 'the treaty was signed in 1648 by both parties.'),
    ('ours, true next', 'the match was delayed by rain.', 'play finally started two hours late.'),
    ('ours, random', 'the match was delayed by rain.', 'photosynthesis turns light into chemical energy.'),
]
log('1. the pre-trained NSP head on single pairs: probability of IsNext')
ex_out = []
for name, a, b in examples:
    # the paper's examples are already split into word pieces ("flight ##less"); join them back before tokenising
    p = is_next_prob(a.replace(' ##', ''), b.replace(' ##', ''))
    ex_out.append(dict(name=name, a=a, b=b, p_isnext=p))
    log(f'   {(f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}"):>7s}  [{name}]  A: {a}  |  B: {b}')

# ---------------------------------------------------------------- 2. 1,000 real pairs
arts = articles('test')
docs = [sum((sentences(p) for p in a), []) for a in arts]
docs = [d for d in docs if len(d) >= 10]
rng = random.Random(0)
pairs = []
for k in range(1000):
    di = rng.randrange(len(docs))
    d = docs[di]
    i = rng.randrange(len(d) - 1)
    if k % 2 == 0:
        pairs.append((d[i], d[i + 1], 0))                          # IsNext
    else:
        other = rng.choice([j for j in range(len(docs)) if j != di])
        pairs.append((d[i], rng.choice(docs[other]), 1))           # NotNext
correct = {0: 0, 1: 0}
for b in range(0, len(pairs), 50):
    chunk = pairs[b:b + 50]
    enc = tok([p[0] for p in chunk], [p[1] for p in chunk], return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        pred = model(**enc).seq_relationship_logits.argmax(-1)
    for p, y in zip(chunk, pred.tolist()):
        correct[p[2]] += int(y == p[2])
acc = (correct[0] + correct[1]) / len(pairs)
log('')
log(f'2. NSP accuracy on {len(pairs)} real sentence pairs from {len(docs)} WikiText-103 test articles (one sentence each side, no masks)')
log(f'   IsNext pairs:  {correct[0]} of 500 right ({correct[0] / 500:.1%})')
log(f'   NotNext pairs: {correct[1]} of 500 right ({correct[1] / 500:.1%})')
log(f'   overall:       {acc:.1%}')

# ---------------------------------------------------------------- 3. footnote 6: raw [CLS] vectors
pairs_sim = [
    ('related', 'A man is playing a guitar on stage.', 'A musician performs a song for the crowd.'),
    ('related', 'The stock market fell sharply today.', 'Share prices dropped a lot this afternoon.'),
    ('unrelated', 'A man is playing a guitar on stage.', 'The stock market fell sharply today.'),
    ('unrelated', 'Penguins cannot fly.', 'The invoice is due next Monday.'),
    ('opposite', 'I loved this movie.', 'I hated this movie.'),
]


def cls_vec(s):
    enc = tok(s, return_tensors='pt')
    with torch.no_grad():
        return model.bert(**enc).last_hidden_state[0, 0]


log('')
log('3. cosine similarity of the raw final [CLS] vectors (no fine-tuning)')
sims = []
for kind, a, b in pairs_sim:
    c = float(torch.cosine_similarity(cls_vec(a), cls_vec(b), dim=0))
    sims.append(dict(kind=kind, a=a, b=b, cos=c))
    log(f'   {c:.3f}  {kind:9s}  {a}  |  {b}')

save('part3_nsp', dict(examples=ex_out, accuracy=dict(n=len(pairs), docs=len(docs), isnext=correct[0] / 500, notnext=correct[1] / 500, all=acc, correct=correct),
                       cls_cosine=sims))
