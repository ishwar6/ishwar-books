"""Part 4: worked numeric examples for every equation of Sections 3.2 and 4, with real numbers.

1. the GLUE head log softmax(C W^T): a toy example (H = 4, K = 3) by hand, then a real SST-2 model
   (textattack/bert-base-uncased-SST-2, BERT-base fine-tuned on SST-2 by its authors, not by us);
2. "self-attention over a packed pair is bidirectional cross attention": how much attention flows from the
   question to the passage and back, layer by layer, in a SQuAD model (csarron/bert-base-uncased-squad-v1);
3. SQuAD v1.1: P_i = softmax(S.T_i), the span score S.T_i + E.T_j with j >= i (toy and real heatmap),
   and the training loss -(log P_start + log P_end);
4. SQuAD v2.0: s_null = S.C + E.C and the threshold tau (deepset/bert-base-uncased-squad2);
5. EM and F1 with the official normalisation, on a few hand-picked answers;
6. SWAG: score_k = V . C_k and a softmax over four choices (toy);
7. how many new weights each head adds, and how many weights fine-tuning updates.
Usage: python bert_part4_math.py   (CPU is fine; about a minute after the downloads)"""
import collections, json, math, re, string
import numpy as np
import torch
from transformers import AutoTokenizer, BertForQuestionAnswering, BertForSequenceClassification
from common import Log, save

torch.manual_seed(0)
log = Log('part4_math')
R = {}
np.set_printoptions(precision=4, suppress=True)
fmt = lambda xs, d=3: '[' + ', '.join(f'{x:.{d}f}' for x in xs) + ']'

# =============== 1a. GLUE head, toy numbers ===============
C = np.array([0.5, -1.0, 2.0, 0.1])                     # C in R^H, H = 4
W = np.array([[1.0, 0.0, 0.5, 0.0],                     # W in R^{K x H}, K = 3 (one row per label)
              [0.0, 1.0, -0.5, 2.0],
              [-0.5, 0.5, 0.0, 1.0]])
z = C @ W.T                                             # logits, one per label
p = np.exp(z) / np.exp(z).sum()
y = 0
log('== 1a. GLUE head, toy: H = 4, K = 3')
log(f'C = {C.tolist()}   (shape {C.shape})')
log(f'W = {W.tolist()}   (shape {W.shape})')
for k in range(3):
    log(f'  logit {k} = C . W[{k}] = ' + ' + '.join(f'({C[h]:g})({W[k, h]:g})' for h in range(4)) + f' = {z[k]:.3f}')
log(f'softmax: exp(logits) = {fmt(np.exp(z))}, sum = {np.exp(z).sum():.3f}, P = {fmt(p)}')
log(f'correct label y = {y}: log softmax = {math.log(p[y]):.3f}, loss = -log P(y) = {-math.log(p[y]):.3f}')
R['glue_toy'] = dict(C=C.tolist(), W=W.tolist(), logits=z.tolist(), p=p.tolist(), y=y, loss=-math.log(p[y]))

# =============== 1b. GLUE head, a real fine-tuned SST-2 model ===============
name = 'textattack/bert-base-uncased-SST-2'
tok = AutoTokenizer.from_pretrained(name)
clf = BertForSequenceClassification.from_pretrained(name).eval()
log(f'\n== 1b. GLUE head, real: {name}')
real = []
for sent, gold in [('a gorgeous, witty, seductive movie.', 1), ('the plot is dull and the acting is worse.', 0)]:
    enc = tok(sent, return_tensors='pt')
    with torch.no_grad():
        out = clf(**enc)
        Cp = clf.bert(**enc).pooler_output[0]               # the (pooled) [CLS] vector, H = 768
    Wr, br = clf.classifier.weight.detach(), clf.classifier.bias.detach()
    zz = Cp @ Wr.T + br
    pp = torch.softmax(zz, -1)
    real.append(dict(sentence=sent, gold=gold, C_first=Cp[:8].tolist(), W0_first=Wr[0, :8].tolist(), W1_first=Wr[1, :8].tolist(),
                     dot=(Cp @ Wr.T).tolist(), bias=br.tolist(), logits=zz.tolist(), lib=out.logits[0].tolist(), p=pp.tolist(),
                     loss=-math.log(pp[gold].item())))
    log(f'"{sent}"  gold label {gold}')
    log(f'  C: shape {tuple(Cp.shape)}, first 8 numbers {fmt(Cp[:8].tolist())}')
    log(f'  W: shape {tuple(Wr.shape)}, bias {fmt(br.tolist())}')
    log(f'  C W^T = {fmt((Cp @ Wr.T).tolist())}, + bias = logits {fmt(zz.tolist())}  (library: {fmt(out.logits[0].tolist())})')
    log(f'  softmax = {fmt(pp.tolist(), 4)};  loss = -log P(gold) = {-math.log(pp[gold].item()):.4f}')
R['glue_real'] = dict(model=name, rows=real)

# =============== 7. how many weights are new, how many are trained ===============
total = sum(p_.numel() for p_ in clf.parameters())
new = clf.classifier.weight.numel() + clf.classifier.bias.numel()
trained = sum(p_.numel() for p_ in clf.parameters() if p_.requires_grad)
log(f'\n== 7. SST-2 model: {total:,} weights in total; {new:,} are new (K x H + K = 2 x 768 + 2); '
    f'{trained:,} have requires_grad = True during fine-tuning (all of them)')
H = 768
heads = {'GLUE, K = 2 (SST-2)': 2 * H + 2, 'GLUE, K = 3 (MNLI)': 3 * H + 3, 'SQuAD (S and E)': 2 * H + 2, 'SWAG (one vector V)': H + 1}
for k, v in heads.items():
    log(f'  new weights, {k}: {v:,}')
R['params'] = dict(total=total, new=new, trained=trained, heads=heads)

# =============== 2. packed pair = cross attention, in a SQuAD model ===============
qa_name = 'csarron/bert-base-uncased-squad-v1'
qtok = AutoTokenizer.from_pretrained(qa_name)
qa = BertForQuestionAnswering.from_pretrained(qa_name, attn_implementation='eager').eval()
q, ctx = 'Where was Ada born?', 'Ada Lovelace was born in London in 1815.'
enc = qtok(q, ctx, return_tensors='pt')
with torch.no_grad():
    out = qa(**enc, output_attentions=True)
toks = qtok.convert_ids_to_tokens(enc.input_ids[0])
seg = enc.token_type_ids[0].tolist()
A = [i for i, t in enumerate(toks) if seg[i] == 0 and t not in ('[CLS]', '[SEP]')]
B = [i for i, t in enumerate(toks) if seg[i] == 1 and t != '[SEP]']
log(f'\n== 2. packed pair in {qa_name}')
log(f'tokens: {toks}')
log(f'segment A (question) positions {A}, segment B (passage) positions {B}')
layers = []
for L, att in enumerate(out.attentions):
    a = att[0].mean(0)                                   # average over the 12 heads: (tokens, tokens), rows sum to 1
    ab = a[A][:, B].sum(1).mean().item()                 # share of a question token's attention that lands on passage tokens
    ba = a[B][:, A].sum(1).mean().item()
    sp = [i for i, t in enumerate(toks) if t in ('[CLS]', '[SEP]')]
    qs = a[A][:, sp].sum(1).mean().item()                # share that lands on [CLS] and the two [SEP]s
    layers.append(dict(layer=L + 1, q_to_p=ab, p_to_q=ba, q_to_special=qs))
    log(f'  layer {L + 1:2d}: question -> passage {ab:.3f}   passage -> question {ba:.3f}   question -> [CLS]/[SEP] {qs:.3f}')
avg = torch.stack([x[0].mean(0) for x in out.attentions]).mean(0)
R['cross'] = dict(model=qa_name, question=q, passage=ctx, tokens=toks, segments=seg, A=A, B=B, layers=layers,
                  avg=avg.tolist(), last=out.attentions[-1][0].mean(0).tolist())
log(f'averaged over all 12 layers and heads, "born" (question) gives {avg[toks.index("born"), B].sum():.3f} of its attention to the passage')

# =============== 3a. span scoring, toy ===============
st = np.array([0.5, 1.0, 3.0, 0.2])                     # S . T_i for 4 passage tokens
en = np.array([2.8, 0.1, 1.0, 1.5])                     # E . T_j
Ps, Pe = np.exp(st) / np.exp(st).sum(), np.exp(en) / np.exp(en).sum()
M = st[:, None] + en[None, :]
free = np.unravel_index(M.argmax(), M.shape)
Mv = np.where(np.triu(np.ones_like(M)) > 0, M, -np.inf)
ok = np.unravel_index(Mv.argmax(), M.shape)
log('\n== 3a. span scoring, toy (4 passage tokens)')
log(f'S.T_i = {st.tolist()}  ->  P_start = {fmt(Ps)}')
log(f'E.T_j = {en.tolist()}  ->  P_end   = {fmt(Pe)}')
log('score(i, j) = S.T_i + E.T_j:')
for i in range(4):
    log('  i=%d: ' % i + '  '.join(('%5.1f' % M[i, j]) + (' ' if j >= i else 'x') for j in range(4)))
log(f'best without the rule: i={free[0]}, j={free[1]} (score {M[free]:.1f}) -> ends before it starts, not allowed')
log(f'best with j >= i:      i={ok[0]}, j={ok[1]} (score {M[ok]:.1f});  P_start*P_end = {Ps[ok[0]] * Pe[ok[1]]:.4f}')
log(f'if the true span is (2, 3): loss = -log {Ps[2]:.4f} - log {Pe[3]:.4f} = {-math.log(Ps[2]) - math.log(Pe[3]):.4f}')
R['span_toy'] = dict(start=st.tolist(), end=en.tolist(), p_start=Ps.tolist(), p_end=Pe.tolist(), M=M.tolist(),
                     free=[int(free[0]), int(free[1])], best=[int(ok[0]), int(ok[1])], loss=-math.log(Ps[2]) - math.log(Pe[3]))

# =============== 3b. span scoring, real heatmap + loss ===============
PASSAGE = ('We introduce a new language representation model called BERT, which stands for Bidirectional Encoder '
           'Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from '
           'unlabeled text by jointly conditioning on both left and right context in all layers. It obtains new '
           'state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score '
           'to 80.5% and SQuAD v1.1 question answering Test F1 to 93.2.')
log(f'\n== 3b. span scoring on the abstract, {qa_name}')
real_spans = []
for question, gold in [('What does BERT stand for?', 'Bidirectional Encoder Representations from Transformers'),
                       ('What does BERT learn from?', 'unlabeled text')]:
    e = qtok(question, PASSAGE, return_tensors='pt', return_offsets_mapping=True)
    off = e.pop('offset_mapping')[0].tolist()
    with torch.no_grad():
        o = qa(**e, output_hidden_states=True)
    s_ = o.start_logits[0].numpy(); e_ = o.end_logits[0].numpy()
    seq = e.sequence_ids(0)
    ctxi = [i for i, s in enumerate(seq) if s == 1]
    T = o.hidden_states[-1][0]
    Sv, Ev = qa.qa_outputs.weight[0].detach(), qa.qa_outputs.weight[1].detach()
    ps = np.exp(s_[ctxi] - s_[ctxi].max()); ps /= ps.sum()
    pe = np.exp(e_[ctxi] - e_[ctxi].max()); pe /= pe.sum()
    gs = PASSAGE.index(gold); ge = gs + len(gold)
    si = next(i for i in ctxi if off[i][0] <= gs < off[i][1])
    ei = next(i for i in ctxi if off[i][0] < ge <= off[i][1])
    loss = -math.log(ps[ctxi.index(si)]) - math.log(pe[ctxi.index(ei)])
    allt = qtok.convert_ids_to_tokens(e.input_ids[0])
    # best span over the whole passage (j >= i, at most 30 tokens)
    best = max((s_[i] + e_[j], i, j) for i in ctxi for j in ctxi if i <= j < i + 30)
    row = dict(question=question, gold=gold, gold_i=si, gold_j=ei, gold_tokens=allt[si:ei + 1], loss=loss,
               p_start_gold=float(ps[ctxi.index(si)]), p_end_gold=float(pe[ctxi.index(ei)]),
               best=[float(best[0]), int(best[1]), int(best[2])], best_text=PASSAGE[off[best[1]][0]:off[best[2]][1]],
               S_first=Sv[:8].tolist(), E_first=Ev[:8].tolist(), S_norm=Sv.norm().item(), E_norm=Ev.norm().item())
    log(f'Q: {question}   gold "{gold}" = tokens {si}..{ei} {allt[si:ei + 1]}')
    log(f'   S, E: shape {tuple(Sv.shape)} each; first 4 of S {fmt(Sv[:4].tolist())}, of E {fmt(Ev[:4].tolist())}')
    log(f'   predicted span "{row["best_text"]}" (tokens {best[1]}..{best[2]}, score {best[0]:.2f})')
    log(f'   P_start(gold start) = {row["p_start_gold"]:.4f}, P_end(gold end) = {row["p_end_gold"]:.4f}, '
        f'loss = -log P_start - log P_end = {loss:.4f}')
    if question.startswith('What does BERT stand'):
        win = [i for i in ctxi if best[1] - 4 <= i <= best[2] + 2]
        H_ = [[float(s_[i] + e_[j]) if j >= i else None for j in win] for i in win]
        row.update(win_tokens=[allt[i] for i in win], heat=H_, win_start=win[0],
                   start_scores=[float(s_[i]) for i in win], end_scores=[float(e_[i]) for i in win],
                   p_start_win=[float(ps[ctxi.index(i)]) for i in win], p_end_win=[float(pe[ctxi.index(i)]) for i in win])
        log(f'   heatmap window: tokens {win[0]}..{win[-1]} {[allt[i] for i in win]}')
        log('   start scores S.T_i: ' + fmt(row['start_scores'], 2))
        log('   end scores   E.T_j: ' + fmt(row['end_scores'], 2))
    real_spans.append(row)
R['span_real'] = real_spans

# =============== 4. SQuAD 2.0: s_null and tau ===============
v2 = 'deepset/bert-base-uncased-squad2'
tok2 = AutoTokenizer.from_pretrained(v2)
m2 = BertForQuestionAnswering.from_pretrained(v2).eval()
best_tau = json.load(open('results/part4_squad.json'))['v2_dev']['best']['tau']
log(f'\n== 4. SQuAD 2.0 no-answer score, {v2}; tau chosen on the dev set = {best_tau:+.2f} (from bert_part4_squad.py)')
nulls = []
for question in ['What does BERT stand for?', 'Who won the football world cup in 2018?', 'How many layers does BERT have?']:
    e = tok2(question, PASSAGE, return_tensors='pt', return_offsets_mapping=True)
    off = e.pop('offset_mapping')[0].tolist()
    with torch.no_grad():
        o = m2(**e)
    s_, e_ = o.start_logits[0].numpy(), o.end_logits[0].numpy()
    ctxi = [i for i, s in enumerate(e.sequence_ids(0)) if s == 1]
    sc, i, j = max((s_[i] + e_[j], i, j) for i in ctxi for j in ctxi if i <= j < i + 30)
    s_null = float(s_[0] + e_[0])
    d = dict(question=question, SC=float(s_[0]), EC=float(e_[0]), s_null=s_null, Si=float(s_[i]), Ej=float(e_[j]),
             best=float(sc), text=PASSAGE[off[i][0]:off[j][1]], diff=float(sc - s_null),
             at0=bool(sc > s_null), at_tau=bool(sc > s_null + best_tau))
    nulls.append(d)
    log(f'Q: {question}')
    log(f'   s_null = S.C + E.C = {d["SC"]:.2f} + {d["EC"]:.2f} = {s_null:.2f}')
    log(f'   best span "{d["text"]}": S.T_i + E.T_j = {d["Si"]:.2f} + {d["Ej"]:.2f} = {sc:.2f}')
    log(f'   tau = 0:     {sc:.2f} > {s_null:.2f} ?  {"yes -> answer" if d["at0"] else "no -> no answer"}')
    log(f'   tau = {best_tau:+.2f}: {sc:.2f} > {s_null + best_tau:.2f} ?  {"yes -> answer" if d["at_tau"] else "no -> no answer"}')
R['null'] = dict(model=v2, tau=best_tau, rows=nulls)


# =============== 5. EM and F1 (official SQuAD normalisation) ===============
def normalize(s):
    s = s.lower()
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())


def prf(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    same = sum((collections.Counter(p) & collections.Counter(g)).values())
    if same == 0:
        return 0.0, 0.0, 0.0, 0
    P, Rc = same / len(p), same / len(g)
    return P, Rc, 2 * P * Rc / (P + Rc), same


log('\n== 5. EM and F1')
emf = []
for pred, gold in [('The Denver Broncos.', 'Denver Broncos'), ('Bidirectional Encoder Representations', 'Bidirectional Encoder Representations from Transformers'),
                   ('the Broncos defeated the Panthers', 'Denver Broncos'), ('eight', '2')]:
    P, Rc, F, same = prf(pred, gold)
    em = float(normalize(pred) == normalize(gold))
    emf.append(dict(pred=pred, gold=gold, pn=normalize(pred), gn=normalize(gold), same=same, P=P, R=Rc, F1=F, EM=em))
    log(f'pred "{pred}" -> "{normalize(pred)}" | gold "{gold}" -> "{normalize(gold)}"')
    log(f'   shared words {same}, precision {P:.3f}, recall {Rc:.3f}, F1 {F:.3f}, EM {em:.0f}')
R['emf1'] = emf

# =============== 6. SWAG, toy ===============
Ck = np.array([[0.9, 0.2, -0.1, 0.4],                   # C_1 .. C_4: the [CLS] vector of each (sentence, ending k) sequence
               [-0.3, 0.8, 0.5, -0.6],
               [0.1, -0.7, 0.9, 0.0],
               [0.4, 0.1, -0.2, -0.9]])
V = np.array([2.0, -1.0, 0.5, 1.0])                     # the one new vector, V in R^H (H = 4 here)
s = Ck @ V
ps = np.exp(s) / np.exp(s).sum()
log('\n== 6. SWAG, toy: four C_k (H = 4), one vector V')
for k in range(4):
    log(f'  s_{k + 1} = V . C_{k + 1} = ' + ' + '.join(f'({V[h]:g})({Ck[k, h]:g})' for h in range(4)) + f' = {s[k]:.2f}')
log(f'softmax over the four = {fmt(ps)}; if ending 1 is right, loss = -log {ps[0]:.3f} = {-math.log(ps[0]):.3f}')
R['swag_toy'] = dict(C=Ck.tolist(), V=V.tolist(), s=s.tolist(), p=ps.tolist(), loss=-math.log(ps[0]))

save('part4_math', R)
