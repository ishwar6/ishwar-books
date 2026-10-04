"""Part 5: the ablation studies, checked with arithmetic and code.

1. Table 5: how much each ablation loses, computed from the paper's own numbers.
2. Table 6: the parameter count of every model size in the table (built on the "meta" device, so no memory is used),
   checked against a formula.
3. A real masked-LM perplexity of bert-base-uncased on WikiText-2 test, with the paper's 15% / 80-10-10 masking.
"""
import math
import torch
from datasets import load_dataset
from transformers import BertConfig, BertModel, BertForMaskedLM, AutoTokenizer
from common import Log, save

log = Log('part5')
out = {}

# ---------------------------------------------------------------- 1. Table 5, typed in from the paper (Dev set)
cols = ['MNLI-m', 'QNLI', 'MRPC', 'SST-2', 'SQuAD']
table5 = {
    'BERT-base': [84.4, 88.4, 86.7, 92.7, 88.5],
    'No NSP': [83.9, 84.9, 86.5, 92.6, 87.9],
    'LTR & No NSP': [82.1, 84.3, 77.5, 92.1, 77.8],
    '+ BiLSTM': [82.1, 84.1, 75.7, 91.6, 84.9],
}
steps = [('No NSP', 'BERT-base', 'cost of removing NSP'),
         ('LTR & No NSP', 'No NSP', 'cost of left-to-right instead of MLM'),
         ('+ BiLSTM', 'LTR & No NSP', 'effect of adding a BiLSTM to the LTR model')]
log('Table 5 (Dev set): change in points')
log(f'{"":46s}' + ''.join(f'{c:>8s}' for c in cols))
out['table5_diffs'] = {}
for a, b, what in steps:
    d = [round(x - y, 1) for x, y in zip(table5[a], table5[b])]
    out['table5_diffs'][f'{a} minus {b}'] = dict(zip(cols, d))
    log(f'{what:46s}' + ''.join(f'{x:+8.1f}' for x in d))
log('')

# ---------------------------------------------------------------- 2. Table 6 parameter counts
def formula(L, H, V=30522, P=512, S=2):
    emb = V * H + P * H + S * H + 2 * H              # token, position, segment embeddings + LayerNorm
    layer = 4 * (H * H + H) + 2 * H + (H * 4 * H + 4 * H) + (4 * H * H + H) + 2 * H   # attention, LN, feed-forward, LN
    pooler = H * H + H
    return emb + L * layer + pooler


table6 = [(3, 768, 12, 5.84, 77.9, 79.8, 88.4), (6, 768, 3, 5.24, 80.6, 82.2, 90.7), (6, 768, 12, 4.68, 81.9, 84.8, 91.3),
          (12, 768, 12, 3.99, 84.4, 86.7, 92.9), (12, 1024, 16, 3.54, 85.7, 86.9, 93.3), (24, 1024, 16, 3.23, 86.6, 87.8, 93.7)]
out['table6'] = []
log('Table 6 model sizes: parameters counted in a real BertModel (built on the meta device)')
log(f'{"#L":>3s} {"#H":>5s} {"#A":>3s} {"params":>13s} {"formula":>13s} {"embeddings":>11s}   ppl  MNLI-m  MRPC  SST-2')
for L, H, A, ppl, mnli, mrpc, sst in table6:
    cfg = BertConfig(vocab_size=30522, hidden_size=H, num_hidden_layers=L, num_attention_heads=A, intermediate_size=4 * H,
                     max_position_embeddings=512, type_vocab_size=2)
    with torch.device('meta'):
        m = BertModel(cfg)
    n = sum(p.numel() for p in m.parameters())
    emb = sum(p.numel() for p in m.embeddings.parameters())
    assert n == formula(L, H), (n, formula(L, H))
    out['table6'].append(dict(L=L, H=H, A=A, params=n, embedding_params=emb, ppl=ppl, mnli=mnli, mrpc=mrpc, sst2=sst))
    log(f'{L:3d} {H:5d} {A:3d} {n:13,d} {formula(L, H):13,d} {emb:11,d}  {ppl:4.2f}  {mnli:6.1f} {mrpc:5.1f} {sst:6.1f}')
log('(includes the 768x768 or 1024x1024 pooler layer; the formula matches the count exactly for every row)')
log('')

# ---------------------------------------------------------------- 3. masked-LM perplexity on real held-out text
torch.manual_seed(0)
g = torch.Generator().manual_seed(0)
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
mlm = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()
raw = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')['test']['text']
# WikiText writes "1 @,@ 000" and "well @-@ known"; undo that, and drop the " = Heading = " lines (BERT's Wikipedia text had no headers)
clean = lambda t: t.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.').strip()
text = '\n'.join(clean(t) for t in raw if t.strip() and not t.strip().startswith('='))
ids = tok(text, add_special_tokens=False).input_ids
L = 510
chunks = [ids[i:i + L] for i in range(0, len(ids) - L + 1, L)]
log(f'WikiText-2 test: {len(ids):,} WordPiece tokens -> {len(chunks)} sequences of 512 ([CLS] + 510 + [SEP])')

nll_sum, n_pred, counts = 0.0, 0, dict(mask=0, random=0, same=0, total=0)
correct = 0
with torch.no_grad():
    for i in range(0, len(chunks), 16):
        x = torch.tensor([[tok.cls_token_id] + c + [tok.sep_token_id] for c in chunks[i:i + 16]])
        labels = x.clone()
        chosen = torch.rand(x.shape, generator=g) < 0.15
        chosen[:, 0] = chosen[:, -1] = False                      # never [CLS] or [SEP]
        r = torch.rand(x.shape, generator=g)
        inp = x.clone()
        inp[chosen & (r < 0.8)] = tok.mask_token_id                                  # 80%: [MASK]
        rnd = chosen & (r >= 0.8) & (r < 0.9)                                        # 10%: a random token
        inp[rnd] = torch.randint(len(tok), x.shape, generator=g)[rnd]
        #                                                                             10%: unchanged
        logits = mlm(input_ids=inp).logits
        lp = torch.log_softmax(logits[chosen], -1)
        nll_sum += -lp.gather(1, labels[chosen][:, None]).sum().item()
        correct += (lp.argmax(-1) == labels[chosen]).sum().item()
        n_pred += int(chosen.sum())
        counts['mask'] += int((chosen & (r < 0.8)).sum()); counts['random'] += int(rnd.sum())
        counts['same'] += int((chosen & (r >= 0.9)).sum()); counts['total'] += x[:, 1:-1].numel()
ppl = math.exp(nll_sum / n_pred)
log(f'predicted positions: {n_pred:,} of {counts["total"]:,} ({100 * n_pred / counts["total"]:.2f}%): '
    f'{counts["mask"]:,} [MASK], {counts["random"]:,} random, {counts["same"]:,} unchanged')
log(f'mean cross-entropy = {nll_sum / n_pred:.4f} nats -> masked-LM perplexity = {ppl:.2f}   (top-1 accuracy {100 * correct / n_pred:.1f}%)')
out['mlm_perplexity'] = dict(data='WikiText-2 raw, test split, @-@ style marks undone, heading lines dropped', model='bert-base-uncased', seq_len=512, n_sequences=len(chunks),
                             n_predicted=n_pred, counts=counts, mean_ce=round(nll_sum / n_pred, 4), perplexity=round(ppl, 2),
                             top1_accuracy=round(100 * correct / n_pred, 1))
save('part5', out)
