"""Part 1, built by hand: the two claims of §1 as code you can read line by line.

  1. Sequential against parallel. A recurrent layer written as a loop (Method 1) and as nn.RNN (Method 2);
     self-attention written as one matrix product (Method 1) and as F.scaled_dot_product_attention (Method 2).
     Then we count the steps that must run in order, and time both on the CPU.
  2. Teacher forcing with the released model (Helsinki-NLP/opus-mt-en-de). We shift the German sentence by hand,
     score all 14 positions in one decoder call, compute the loss by hand, and then write the same sentence with
     our own greedy loop, one decoder call per token, against model.generate.
Run: ~/.venvs/attn/bin/python tx_part1_scratch.py   (CPU, under a minute after the download)"""
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import Log, save

log = Log('part1_scratch')
torch.manual_seed(0)
torch.set_num_threads(4)
R = {}


def diff(a, b):
    return (a - b).abs().max().item()


# ------------------------------------------------------------------ 1. the loop and the matrix product
def rnn_layer(X, W, U, b):
    """Method 1: a recurrent layer. X is (batch, n, d). Step t cannot start before step t - 1 ends."""
    B, n, d = X.shape
    h = torch.zeros(B, d)                               # h_0
    states = []
    for t in range(n):                                  # n steps, one after the other
        h = torch.tanh(X[:, t] @ W.T + h @ U.T + b)     # (B, d): needs the h of the step before
        states.append(h)
    return torch.stack(states, 1)                       # (B, n, d)


def attention_layer(X):
    """Method 1: self-attention with no learned weights, Q = K = V = X. All n rows in one go."""
    d = X.shape[-1]
    scores = X @ X.transpose(1, 2) / math.sqrt(d)      # (B, n, n): every word against every word
    weights = torch.softmax(scores, dim=-1)             # each row adds up to 1
    return weights @ X                                  # (B, n, d): a weighted mix for every word at once


log('== 1. Sequential against parallel (width d = 512, batch 32)')
B, n, d = 32, 12, 512
X = torch.randn(B, n, d)
rnn = nn.RNN(d, d, batch_first=True)                    # Method 2 for the loop
W, U, b = rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0 + rnn.bias_hh_l0
with torch.no_grad():
    H_ours = rnn_layer(X, W, U, b)
    H_lib, _ = rnn(X)
    A_ours = attention_layer(X)
    A_lib = F.scaled_dot_product_attention(X, X, X)     # Method 2 for attention (scale 1/sqrt(d) by default)
log(f'  X {tuple(X.shape)}')
log(f'  RNN loop: {n} steps in order -> {tuple(H_ours.shape)}; max |ours - nn.RNN| = {diff(H_ours, H_lib):.1e}')
log(f'  attention: 1 product X X^T {(B, n, n)}, softmax, 1 product with X -> {tuple(A_ours.shape)}; '
    f'max |ours - F.scaled_dot_product_attention| = {diff(A_ours, A_lib):.1e}')

# Each attention row needs only X, not the other outputs: compute row 7 alone.
with torch.no_grad():
    row7 = torch.softmax(X[:, 7:8] @ X.transpose(1, 2) / math.sqrt(d), -1) @ X     # (B, 1, d)
log(f'  attention row 7 computed alone, from X only: max |row 7 alone - row 7 of the full product| = {diff(row7[:, 0], A_ours[:, 7]):.1e}')

# Change word 3 only, and see which outputs move. Small word vectors keep the attention weights spread out.
with torch.no_grad():
    Xs = 0.05 * torch.randn(B, n, d)                    # (B, 12, 512)
    Xs2 = Xs.clone(); Xs2[:, 3] += 0.05                 # the same sentence, word 3 changed
    move_rnn = (rnn_layer(Xs2, W, U, b) - rnn_layer(Xs, W, U, b)).abs().amax(dim=(0, 2))   # (12,): one number per output
    move_att = (attention_layer(Xs2) - attention_layer(Xs)).abs().amax(dim=(0, 2))
log('  change word 3 only; largest change of each output 0..11:')
log('    RNN       ' + ' '.join(f'{v:.3f}' for v in move_rnn))
log('    attention ' + ' '.join(f'{v:.3f}' for v in move_att))


def fastest(f, k=10):
    f()
    best = 1e9
    for _ in range(k):
        t0 = time.perf_counter(); f(); best = min(best, time.perf_counter() - t0)
    return best * 1000


log('')
log('   time on the CPU (4 threads), forward pass only, fastest of 10 runs, random weights')
log(f'{"n":>6} {"steps in order":>15} {"our loop ms":>12} {"nn.RNN ms":>10} {"our attention ms":>17} {"SDPA ms":>8}')
rows = []
with torch.no_grad():
    for n in [16, 64, 256]:
        X = torch.randn(B, n, d)
        t_loop = fastest(lambda: rnn_layer(X, W, U, b))
        t_rnn = fastest(lambda: rnn(X))
        t_att = fastest(lambda: attention_layer(X))
        t_sdpa = fastest(lambda: F.scaled_dot_product_attention(X, X, X))
        rows.append(dict(n=n, loop=t_loop, rnn=t_rnn, att=t_att, sdpa=t_sdpa))
        log(f'{n:>6} {f"{n} vs 1":>15} {t_loop:>12.1f} {t_rnn:>10.1f} {t_att:>17.1f} {t_sdpa:>8.1f}')
R['timing'] = rows

# ------------------------------------------------------------------ 2. teacher forcing with the released model
from transformers import MarianMTModel, MarianTokenizer
from transformers.models.marian.modeling_marian import shift_tokens_right
NAME = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME).eval()
cfg = model.config
src = tok(['The dog sat on the mat because it was tired.'], return_tensors='pt')
labels = tok(text_target=['Der Hund saß auf der Matte, weil er müde war.'], return_tensors='pt')['input_ids']
m = labels.shape[1]

log('')
log('== 2. Teacher forcing with ' + NAME)
log(f'  English ids {tuple(src.input_ids.shape)}; German labels {tuple(labels.shape)}: {tok.convert_ids_to_tokens(labels[0])}')

# Step 1: the decoder input is the German sentence moved one place to the right, with a start token in front.
start = torch.tensor([[cfg.decoder_start_token_id]])          # Marian uses <pad> (id 58100) as the start token
dec_in = torch.cat([start, labels[:, :-1]], dim=1)            # (1, m): drop the last token, put start in front
dec_lib = shift_tokens_right(labels, cfg.pad_token_id, cfg.decoder_start_token_id)
log(f'  decoder input {tuple(dec_in.shape)}: {tok.convert_ids_to_tokens(dec_in[0])}')
log(f'  same as the library shift_tokens_right: {torch.equal(dec_in, dec_lib)}')

# Step 2: one decoder call scores every position. The causal mask inside stops position t reading t + 1, ...
with torch.no_grad():
    out = model(**src, decoder_input_ids=dec_in)
logits = out.logits                                           # (1, m, 58101): one row of scores per position
log(f'  one call: logits {tuple(logits.shape)}')

# Step 3: the loss by hand: -log p(correct token), averaged over the m positions.
logp = torch.log_softmax(logits, dim=-1)                      # (1, m, 58101)
logp_right = logp[0, torch.arange(m), labels[0]]              # (m,): log p of the right token at each position
loss_ours = -logp_right.mean()
with torch.no_grad():
    loss_lib = model(**src, labels=labels).loss               # Method 2: the library shifts and averages for us
log(f'  loss by hand {loss_ours:.4f}; library loss {loss_lib:.4f}; difference {abs(loss_ours - loss_lib):.1e}')
log(f'  {"pos":>3}  {"decoder reads":>14}  {"must predict":>13}  {"p(right)":>8}  {"top guess":>10}')
pred = logits.argmax(-1)[0]
for t in range(m):
    log(f'  {t:>3}  {tok.convert_ids_to_tokens([int(dec_in[0, t])])[0]:>14}  {tok.convert_ids_to_tokens([int(labels[0, t])])[0]:>13}'
        f'  {logp_right[t].exp():8.3f}  {tok.convert_ids_to_tokens([int(pred[t])])[0]:>10}')
log(f'  top guess equals the label at {int((pred == labels[0]).sum())} of {m} positions')
R['tf'] = dict(m=m, loss=loss_ours.item(), loss_lib=loss_lib.item(), p=logp_right.exp().tolist())


# ------------------------------------------------------------------ 3. writing: one call per token
def greedy(model, src, max_new=40):
    """Method 1: our own generation loop. The encoder runs once; the decoder runs once per new token."""
    enc_out = model.get_encoder()(**src)                      # (1, 12, 512), reused at every step
    ys = torch.tensor([[cfg.decoder_start_token_id]])         # (1, 1): only the start token
    step_logits = []
    for _ in range(max_new):
        logits = model(encoder_outputs=enc_out, attention_mask=src.attention_mask, decoder_input_ids=ys).logits
        last = logits[:, -1].clone()                          # (1, 58101): scores for the next token only
        last[:, cfg.pad_token_id] = -float('inf')             # never write the start/pad token (generate does the same)
        nxt = last.argmax(-1, keepdim=True)                   # (1, 1): the most likely token
        step_logits.append(logits[0, -1])
        ys = torch.cat([ys, nxt], dim=1)                      # feed it back in: the input grows by one
        if nxt.item() == cfg.eos_token_id:
            break
    return ys, torch.stack(step_logits)


with torch.no_grad():
    ys, step_logits = greedy(model, src)
    gen = model.generate(**src, num_beams=1, do_sample=False, max_new_tokens=40)   # Method 2
log('')
log('== 3. Writing the translation: our greedy loop against model.generate')
log(f'  ours    : {tok.decode(ys[0], skip_special_tokens=True)}  ({ys.shape[1] - 1} decoder calls)')
log(f'  generate: {tok.decode(gen[0], skip_special_tokens=True)}')
log(f'  same tokens: {torch.equal(ys, gen)}')
log(f'  max |teacher-forced logits - loop logits at the same position| = {diff(logits[0], step_logits):.1e}')
R['greedy'] = dict(calls=ys.shape[1] - 1, same=torch.equal(ys, gen))
save('part1_scratch', R)
