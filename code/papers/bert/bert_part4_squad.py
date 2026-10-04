"""Part 4: question answering with BERT, the way Sections 4.2 and 4.3 of the paper describe it.

We did NOT fine-tune these models. We use two public BERT-base checkpoints from the Hugging Face hub:
  csarron/bert-base-uncased-squad-v1   (fine-tuned on SQuAD v1.1 by its author)
  deepset/bert-base-uncased-squad2     (fine-tuned on SQuAD v2.0 by its author)
What we check ourselves:
  1. the start/end layer is exactly two vectors S and E, and S.T_i is the start score of token i;
  2. our own span search (max S.T_i + E.T_j with j >= i) on a passage;
  3. EM and F1 on the full SQuAD v1.1 dev set, with our own decoding and the official answer normalisation;
  4. SQuAD v2.0: the no-answer score s_null = S.C + E.C, and the threshold tau chosen on the dev set.
Usage: python bert_part4_squad.py   (Apple GPU if available; about 10 minutes)"""
import collections, re, string, time
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BertForQuestionAnswering
from common import Log, save

dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
log = Log('part4_squad')
MAX_LEN, STRIDE, MAX_ANSWER, NBEST = 384, 128, 30, 20   # the settings of the released run_squad.py

PASSAGE = ('We introduce a new language representation model called BERT, which stands for Bidirectional Encoder '
           'Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from '
           'unlabeled text by jointly conditioning on both left and right context in all layers. It obtains new '
           'state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score '
           'to 80.5% and SQuAD v1.1 question answering Test F1 to 93.2.')
QUESTIONS = ['What does BERT stand for?', 'What is the GLUE score of BERT?', 'On how many tasks does BERT get new results?',
             'What does BERT learn from?']
UNANSWERABLE = ['Who won the football world cup in 2018?', 'How many layers does BERT have?']


# ---------- official SQuAD answer normalisation and scores ----------
def normalize(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = collections.Counter(p) & collections.Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def em(pred, gold):
    return float(normalize(pred) == normalize(gold))


# ---------- the model pieces ----------
def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = BertForQuestionAnswering.from_pretrained(name).to(dev).eval()
    return tok, model


def features(tok, questions, contexts):
    return tok(questions, contexts, truncation='only_second', max_length=MAX_LEN, stride=STRIDE,
               return_overflowing_tokens=True, return_offsets_mapping=True, padding='max_length')


@torch.no_grad()
def logits(model, enc, bs=32):
    S, E = [], []
    n = len(enc['input_ids'])
    for k in range(0, n, bs):
        batch = {key: torch.tensor(enc[key][k:k + bs]).to(dev) for key in ('input_ids', 'token_type_ids', 'attention_mask')}
        out = model(**batch)
        S.append(out.start_logits.float().cpu().numpy()); E.append(out.end_logits.float().cpu().numpy())
        if (k // bs) % 50 == 0:
            print(f'  windows {k}/{n}', flush=True)
    return np.concatenate(S), np.concatenate(E)


def best_spans(start, end, offsets, seq_ids, k=NBEST):
    """Candidate spans (score, i, j): S.T_i + E.T_j, j >= i, both inside the passage, at most MAX_ANSWER tokens."""
    ctx = [i for i, s in enumerate(seq_ids) if s == 1]
    lo, hi = ctx[0], ctx[-1]
    si = np.argsort(start)[::-1][:k]
    ei = np.argsort(end)[::-1][:k]
    out = []
    for i in si:
        for j in ei:
            if lo <= i <= hi and lo <= j <= hi and j >= i and j - i + 1 <= MAX_ANSWER:
                out.append((float(start[i] + end[j]), int(i), int(j)))
    return sorted(out, reverse=True)


# =============== 1 and 2: SQuAD v1.1 model on one passage ===============
tok, model = load('csarron/bert-base-uncased-squad-v1')
W, b = model.qa_outputs.weight.detach().cpu(), model.qa_outputs.bias.detach().cpu()
log(f'SQuAD v1.1 model: the output layer is {tuple(W.shape)} -> row 0 is the start vector S, row 1 the end vector E (H = {W.shape[1]})')
log(f'  it also has a bias: start {b[0].item():+.4f}, end {b[1].item():+.4f} (a constant added to every position, so it cancels in the softmax)')

demo = []
for q in QUESTIONS:
    enc = tok(q, PASSAGE, return_tensors='pt', return_offsets_mapping=True)
    off = enc.pop('offset_mapping')[0].tolist()
    with torch.no_grad():
        out = model(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True)
    T = out.hidden_states[-1][0].cpu()                         # T_i: the final vector of every token, (tokens, 768)
    s_by_hand = T @ W[0] + b[0]                                 # S . T_i (+ bias)
    gap = (s_by_hand - out.start_logits[0].cpu()).abs().max().item()
    seq = enc.sequence_ids(0)
    ctx = [i for i, s in enumerate(seq) if s == 1]
    st, en = out.start_logits[0].cpu().numpy(), out.end_logits[0].cpu().numpy()
    P = np.exp(st[ctx] - st[ctx].max()); P /= P.sum()           # P_i: softmax over the passage tokens only
    P_nobias = torch.softmax((T @ W[0])[ctx], 0).numpy()        # the same without the bias
    spans = best_spans(st, en, off, seq)
    score, i, j = spans[0]
    ans = PASSAGE[off[i][0]:off[j][1]]
    toks = tok.convert_ids_to_tokens(enc['input_ids'][0])
    top = sorted(zip(P.tolist(), [toks[c] for c in ctx]), reverse=True)[:3]
    demo.append(dict(question=q, answer=ans, score=score, i=i, j=j, max_gap=gap, bias_effect=float(np.abs(P - P_nobias).max()),
                     tokens=[toks[c] for c in ctx], p_start=P.tolist(), p_end=(lambda e: (e / e.sum()).tolist())(np.exp(en[ctx] - en[ctx].max())),
                     top_start=top, top_spans=[(sc, PASSAGE[off[a][0]:off[c][1]]) for sc, a, c in spans[:3]]))
    log(f'Q: {q}')
    log(f'   answer: "{ans}"  (tokens {i}..{j}, span score S.T_i + E.T_j = {score:.2f})')
    log(f'   runner-up spans: ' + '; '.join(f'"{t}" {sc:.2f}' for sc, t in demo[-1]['top_spans'][1:]))
    log(f'   most likely start tokens: ' + ', '.join(f'{t} {p:.3f}' for p, t in top))
    log(f'   check: |S.T_i by hand - model start logit| max = {gap:.1e}; softmax with vs without bias differs by {demo[-1]["bias_effect"]:.1e}')

# =============== 3: full SQuAD v1.1 dev set ===============
sq = load_dataset('rajpurkar/squad', split='validation')
t0 = time.time()
qs, cs = list(sq['question']), list(sq['context'])
enc = features(tok, [q.lstrip() for q in qs], cs)
st, en = logits(model, enc)
by_ex = collections.defaultdict(list)
for f, ex in enumerate(enc['overflow_to_sample_mapping']):
    by_ex[ex].append(f)
EM = F1 = 0.0
answers = list(sq['answers'])
wrong = []
for ex, feats in by_ex.items():
    cands = []
    for f in feats:
        off = enc['offset_mapping'][f]
        for sc, i, j in best_spans(st[f], en[f], off, enc.sequence_ids(f)):
            cands.append((sc, cs[ex][off[i][0]:off[j][1]]))
    pred = max(cands)[1] if cands else ''
    golds = answers[ex]['text']
    e, f_ = max(em(pred, g) for g in golds), max(f1(pred, g) for g in golds)
    EM += e; F1 += f_
    if f_ == 0 and len(wrong) < 3:
        wrong.append(dict(question=qs[ex], pred=pred, gold=golds[0]))
n = len(by_ex)
EM, F1 = 100 * EM / n, 100 * F1 / n
mins1 = (time.time() - t0) / 60
log(f'SQuAD v1.1 dev: {n} questions ({len(st)} windows of {MAX_LEN} tokens, stride {STRIDE}): EM {EM:.2f}, F1 {F1:.2f}  ({mins1:.1f} min)')
for w in wrong:
    log(f'   a miss: Q "{w["question"]}" predicted "{w["pred"]}", gold "{w["gold"]}"')

# =============== 4: SQuAD v2.0, the no-answer score and the threshold tau ===============
tok2, model2 = load('deepset/bert-base-uncased-squad2')
demo2 = []
for q in QUESTIONS[:2] + UNANSWERABLE:
    e2 = tok2(q, PASSAGE, return_tensors='pt', return_offsets_mapping=True)
    off = e2.pop('offset_mapping')[0].tolist()
    with torch.no_grad():
        o = model2(**{k: v.to(dev) for k, v in e2.items()})
    s_, e_ = o.start_logits[0].cpu().numpy(), o.end_logits[0].cpu().numpy()
    s_null = float(s_[0] + e_[0])                              # S.C + E.C: the "span" that starts and ends at [CLS]
    sc, i, j = best_spans(s_, e_, off, e2.sequence_ids(0))[0]
    demo2.append(dict(question=q, s_null=s_null, best=sc, best_text=PASSAGE[off[i][0]:off[j][1]], diff=sc - s_null))
    log(f'SQuAD 2.0 Q: {q}')
    log(f'   s_null = S.C + E.C = {s_null:.2f};  best span "{demo2[-1]["best_text"]}" scores {sc:.2f};  difference {sc - s_null:+.2f}')

sq2 = load_dataset('rajpurkar/squad_v2', split='validation')
t0 = time.time()
qs2, cs2, ans2 = list(sq2['question']), list(sq2['context']), list(sq2['answers'])
enc2 = features(tok2, [q.lstrip() for q in qs2], cs2)
st2, en2 = logits(model2, enc2)
by_ex2 = collections.defaultdict(list)
for f, ex in enumerate(enc2['overflow_to_sample_mapping']):
    by_ex2[ex].append(f)
rows = []   # per question: (best non-null score - null score, F1 if we answer, F1 if we abstain, EM if answer, EM if abstain)
for ex, feats in by_ex2.items():
    cands, nulls = [], []
    for f in feats:
        off = enc2['offset_mapping'][f]
        nulls.append(st2[f][0] + en2[f][0])
        for sc, i, j in best_spans(st2[f], en2[f], off, enc2.sequence_ids(f)):
            cands.append((sc, cs2[ex][off[i][0]:off[j][1]]))
    best, text = max(cands) if cands else (-1e9, '')
    golds = ans2[ex]['text'] or ['']
    rows.append((best - min(nulls), max(f1(text, g) for g in golds), float(golds == ['']),
                 max(em(text, g) for g in golds), float(golds == [''])))
rows = np.array(rows)
taus = np.round(np.arange(-6, 6.01, 0.25), 2)
sweep = []
for tau in taus:
    answer = rows[:, 0] > tau                                  # predict a span only when s_hat > s_null + tau
    f1s = np.where(answer, rows[:, 1], rows[:, 2]).mean() * 100
    ems = np.where(answer, rows[:, 3], rows[:, 4]).mean() * 100
    sweep.append(dict(tau=float(tau), f1=float(f1s), em=float(ems), answered=float(answer.mean())))
best_tau = max(sweep, key=lambda r: r['f1'])
at0 = next(r for r in sweep if r['tau'] == 0)
mins2 = (time.time() - t0) / 60
log(f'SQuAD v2.0 dev: {len(rows)} questions, {int((rows[:, 2] == 1).sum())} have no answer')
log(f'   tau = 0:    F1 {at0["f1"]:.2f}, EM {at0["em"]:.2f}, answers {100 * at0["answered"]:.1f}% of questions')
log(f'   best tau = {best_tau["tau"]:+.2f}: F1 {best_tau["f1"]:.2f}, EM {best_tau["em"]:.2f}, answers {100 * best_tau["answered"]:.1f}% of questions  ({mins2:.1f} min)')

save('part4_squad', dict(v1_model='csarron/bert-base-uncased-squad-v1', v2_model='deepset/bert-base-uncased-squad2',
                         max_len=MAX_LEN, stride=STRIDE, max_answer=MAX_ANSWER, demo=demo, v1_dev=dict(n=n, em=EM, f1=F1, minutes=mins1, misses=wrong),
                         demo2=demo2, v2_dev=dict(n=len(rows), n_noans=int((rows[:, 2] == 1).sum()), at0=at0, best=best_tau, sweep=sweep, minutes=mins2)))
