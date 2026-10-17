"""Part 7: our from-scratch Transformer with the weights of Helsinki-NLP/opus-mt-en-de against transformers' MarianMTModel.

1. Copy every tensor (scratch/load_marian.py) and report what was copied and why some were not.
2. Encoder output of both models for the same sentences: max abs difference.
3. Decoder logits for the same German prefix (teacher forcing): max abs difference.
4. Greedy translation of several sentences: our greedy_decode against MarianMTModel.generate(num_beams=1).
5. Our beam search (beam 4) on the same sentences.
Writes results/part7_marian_stdout.txt and results/part7_marian.json. CPU, about half a minute."""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'scratch'))
from common import Log, save
from model import greedy_decode, beam_search, count_parameters
from load_marian import load_marian

log = Log('part7_marian')
R = {}

model, tok, info, hf = load_marian(return_hf=True)
log('== 1. Copy the weights')
log(f'Marian tensors: {info["marian_tensors"]}, copied into our model: {info["copied_tensors"]}')
log('not copied: encoder/decoder embed_tokens and lm_head (the same table as model.shared),')
log('            encoder/decoder embed_positions (fixed sinusoids, not learned)')
log(f'our sin_then_cos table vs Marian\'s position table, max abs difference: {info["position_table_max_diff"]:.1e}')
log(f'parameters: ours {count_parameters(model):,}, Marian {sum(p.numel() for p in hf.parameters()):,}')
gap = sum(p.numel() for p in hf.parameters()) - count_parameters(model)
log(f'gap: {gap:,} = 2 x 512 x 512, the two fixed position tables that Marian stores as (frozen) parameters')
R['info'] = info

sentences = ['The cat sat on the mat.',
             'Attention is all you need.',
             'The weather is nice today, so we will walk to the station.',
             'I have read this book twice.',
             'Neural networks learn from examples.']
batch = tok(sentences, return_tensors='pt', padding=True)
src = batch.input_ids                                        # (5, n), padded with id 58100

log('')
log('== 2. Encoder output')
with torch.no_grad():
    ours_mem, src_mask = model.encode(src)                   # (5, n, 512)
    hf_mem = hf.model.encoder(input_ids=src, attention_mask=batch.attention_mask).last_hidden_state
keep = batch.attention_mask.bool()                           # compare real tokens only (pad rows are never read)
d_enc = (ours_mem - hf_mem)[keep].abs().max().item()
log(f'src {tuple(src.shape)} -> memory {tuple(ours_mem.shape)}; max abs difference on real tokens: {d_enc:.2e}')
log(f'size of the numbers: max abs value {hf_mem[keep].abs().max().item():.2f}')

log('')
log('== 3. Decoder logits (teacher forcing with Marian\'s own German translation)')
with torch.no_grad():
    ref = hf.generate(**batch, num_beams=1, do_sample=False, max_new_tokens=64)   # (5, t), starts with 58100
    ours_logits = model.decode(ref, ours_mem, src_mask)                          # (5, t, 58101)
    hf_logits = hf(input_ids=src, attention_mask=batch.attention_mask, decoder_input_ids=ref).logits
tkeep = ref != tok.pad_token_id
tkeep[:, 0] = True                                           # position 0 is the start token (it uses the pad id)
d_log = (ours_logits - hf_logits)[tkeep].abs().max().item()
log(f'target {tuple(ref.shape)} -> logits {tuple(ours_logits.shape)}; max abs difference: {d_log:.2e}')
same_argmax = (ours_logits.argmax(-1) == hf_logits.argmax(-1))[tkeep].float().mean().item()
log(f'same top token at {same_argmax * 100:.1f}% of {int(tkeep.sum())} positions')
R['enc_diff'], R['logit_diff'], R['same_argmax'] = d_enc, d_log, same_argmax

log('')
log('== 4. Greedy translation: ours vs MarianMTModel.generate(num_beams=1)')
t0 = time.time()
ours_ids = greedy_decode(model, src, info['start_id'], info['eos_id'], max_len=64)
t_ours = time.time() - t0
ours_txt = tok.batch_decode(ours_ids, skip_special_tokens=True)
hf_txt = tok.batch_decode(ref, skip_special_tokens=True)
same = 0
for s, a, b in zip(sentences, ours_txt, hf_txt):
    log(f'EN    {s}')
    log(f'ours  {a}')
    log(f'Marian {b}' if a != b else '        (same as MarianMTModel)')
    same += a == b
log(f'identical: {same} of {len(sentences)}')
R['greedy'] = [{'en': s, 'ours': a, 'marian': b} for s, a, b in zip(sentences, ours_txt, hf_txt)]

log('')
log('== 5. Our beam search (beam 4, length penalty alpha 0.6, as in §6.1)')
for s in sentences[2:4]:
    ids = tok([s], return_tensors='pt').input_ids
    out = beam_search(model, ids, info['start_id'], info['eos_id'], beam_size=4, max_len=64, alpha=0.6)
    log(f'EN    {s}')
    log(f'beam  {tok.decode(out, skip_special_tokens=True)}')
save('part7_marian', R)
