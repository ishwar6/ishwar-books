"""Part 4: the task heads of Section 4, written by hand and checked against the library versions.

1. GLUE (Section 4.1): a classification layer W (K x H) on the [CLS] vector, loss = -log softmax(C W^T)[label].
   The paper writes C = final hidden vector of [CLS]. The released code (and Hugging Face) first passes it
   through the "pooler" (a dense layer + tanh, trained during pre-training) and feeds that to W. We check both.
2. SWAG (Section 4.4): four (sentence, ending) sequences, one vector w, score = C . w, softmax over the four.
The heads here are untrained (random W and w): this shows the shapes and the maths, not accuracy.
Runs on the CPU in a few seconds."""
import torch
from transformers import AutoTokenizer, BertForSequenceClassification, BertForMultipleChoice
from common import Log, save

torch.manual_seed(0)
log = Log('part4_heads')
tok = AutoTokenizer.from_pretrained('bert-base-uncased')

# ---------- 1. sentence-pair classification, MNLI style: K = 3 labels ----------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3).eval()
enc = tok('A man is playing a guitar.', 'A person is making music.', return_tensors='pt')
label = torch.tensor([0])                                   # say 0 = entailment
with torch.no_grad():
    out = model(**enc, labels=label, output_hidden_states=True)
    C = out.hidden_states[-1][:, 0]                         # final hidden vector of [CLS]: shape (1, H)
    pooled = torch.tanh(model.bert.pooler.dense(C))         # what the released code feeds to the classifier
    W, bias = model.classifier.weight, model.classifier.bias   # W: (K, H)
    logits_ours = pooled @ W.T + bias                       # C W^T (+ bias), one score per label
    loss_ours = -torch.log_softmax(logits_ours, -1)[0, label]
    logits_raw = C @ W.T + bias                             # the formula exactly as printed, on the raw C
log(f'input tokens: {tok.convert_ids_to_tokens(enc.input_ids[0])}')
log(f'C (final [CLS] vector): shape {tuple(C.shape)};  W: shape {tuple(W.shape)}  (K = 3 labels, H = 768)')
log(f'logits from the library:  {[round(v, 4) for v in out.logits[0].tolist()]}')
log(f'logits by hand (pooled C): {[round(v, 4) for v in logits_ours[0].tolist()]}')
log(f'loss from the library {out.loss.item():.6f}  vs  -log softmax(C W^T)[label] by hand {loss_ours.item():.6f}')
log(f'probabilities (untrained W, so they mean nothing yet): {[round(v, 3) for v in torch.softmax(logits_ours, -1)[0].tolist()]}')
log(f'with the raw C instead of the pooled C the logits would be {[round(v, 4) for v in logits_raw[0].tolist()]} (the library uses the pooled C)')
new = model.classifier.weight.numel() + model.classifier.bias.numel()
total = sum(p.numel() for p in model.parameters())
log(f'new parameters for this task: {new:,} out of {total:,} ({100 * new / total:.4f}%)')
gap = (out.logits - logits_ours).abs().max().item()

# ---------- 2. SWAG: one score per (sentence, ending) pair, softmax over 4 ----------
mc = BertForMultipleChoice.from_pretrained('bert-base-uncased').eval()
ctx = 'She opened the fridge and'
endings = ['took out a carton of milk.', 'flew to the moon.', 'the fridge sang a song.', 'painted the ocean blue.']
e = tok([ctx] * 4, endings, return_tensors='pt', padding=True)
e = {k: v.unsqueeze(0) for k, v in e.items()}              # (batch 1, 4 choices, tokens)
with torch.no_grad():
    o = mc(**e)
    flat = {k: v.view(4, -1) for k, v in e.items()}
    pooled4 = mc.bert(**flat).pooler_output                 # (4, H): one C per choice
    w = mc.classifier.weight                                # (1, H): the single new vector
    scores = (pooled4 @ w.T).view(1, 4) + mc.classifier.bias
log(f'SWAG input: {tuple(e["input_ids"].shape)} = (batch, 4 choices, tokens); the vector w: {tuple(w.shape)}')
log(f'scores C.w by hand: {[round(v, 4) for v in scores[0].tolist()]};  library: {[round(v, 4) for v in o.logits[0].tolist()]}')
log(f'softmax over the 4 choices (untrained w): {[round(v, 3) for v in torch.softmax(scores, -1)[0].tolist()]}')
save('part4_heads', dict(glue_logits_lib=out.logits[0].tolist(), glue_logits_ours=logits_ours[0].tolist(), loss_lib=out.loss.item(),
                         loss_ours=loss_ours.item(), max_logit_gap=gap, new_params=new, total_params=total,
                         swag_scores=scores[0].tolist(), swag_lib=o.logits[0].tolist()))
