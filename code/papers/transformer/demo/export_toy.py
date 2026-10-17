"""Export the series' own from-scratch Transformer (code/papers/transformer/scratch/model.py, trained by train_toy.py on
"2021-03-12" -> "12 March 2021") to two ONNX files that the browser runs with onnxruntime-web.

  encoder.onnx   src_ids (1, n) int64, pe (n, d_model) float32
                 -> emb (1, n, d_model)      token embedding times sqrt(d_model)            (§3.4)
                    x0 (1, n, d_model)       emb + pe, the input of encoder layer 1          (§3.5)
                    memory (1, n, d_model)   the encoder output
                    enc_att (L, H, n, n)     the self-attention weights of every layer and head
  decoder.onnx   tgt_ids (1, t) int64, memory (1, n, d_model), src_ids (1, n) int64, pe (t, d_model), mask (t, t)
                 -> logits (1, t, vocab)     row k scores the token that comes after token k
                    self_att (L, H, t, t)    masked self-attention weights
                    cross_att (L, H, t, n)   encoder-decoder attention weights

The position table and the decoder mask are inputs, so the page can feed a table of zeros (no positions) or a mask of
ones (no mask) and show what breaks. With the normal table and the causal mask, the two graphs compute exactly what
model.py computes: the wrappers below call the same modules in the same order (dropout is off in eval mode).

Checks (written to results/toy_export.json):
  - onnxruntime vs PyTorch (model.forward) logits on held-out dates, teacher forced: max |difference|;
  - greedy decoding through the two ONNX graphs vs model.greedy_decode: same output text;
  - exact match on the 2,000 held-out dates with: the normal model, no position encoding, no decoder mask.
Run from anywhere:  python export_toy.py   (needs toy.pt from scratch/train_toy.py; onnx and onnxruntime)
"""
import json, os, shutil, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(HERE, '..', 'scratch')
sys.path.insert(0, SCRATCH)
from model import Config, Transformer, greedy_decode, padding_mask, sinusoid_table, count_parameters  # noqa: E402
import toy_task as task  # noqa: E402

PUB = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'transformer', 'toy')
RES = os.path.join(HERE, 'results')
os.makedirs(PUB, exist_ok=True)
os.makedirs(RES, exist_ok=True)


class EncoderStep(torch.nn.Module):
    def __init__(self, m: Transformer):
        super().__init__()
        self.m = m

    def forward(self, src_ids, pe):
        enc = self.m.encoder
        src_mask = padding_mask(src_ids, self.m.cfg.pad_id)
        emb = enc.embed(src_ids)                       # (b, n, d): table lookup times sqrt(d_model)
        x0 = emb + pe.unsqueeze(0)                     # what PositionalEncoding adds, from the input table
        x = x0
        att = []
        for layer in enc.layers:
            x = layer(x, src_mask)
            att.append(layer.self_attn.last_weights[0])   # (H, n, n) of the first sentence
        return emb, x0, x, torch.stack(att)


class DecoderStep(torch.nn.Module):
    def __init__(self, m: Transformer):
        super().__init__()
        self.m = m

    def forward(self, tgt_ids, memory, src_ids, pe, mask):
        dec = self.m.decoder
        src_mask = padding_mask(src_ids, self.m.cfg.pad_id)
        tgt_mask = (mask > 0.5).unsqueeze(0).unsqueeze(0) & padding_mask(tgt_ids, self.m.cfg.pad_id)
        y = dec.embed(tgt_ids) + pe.unsqueeze(0)
        sa, ca = [], []
        for layer in dec.layers:
            y = layer(y, memory, tgt_mask, src_mask)
            sa.append(layer.self_attn.last_weights[0])
            ca.append(layer.cross_attn.last_weights[0])
        return self.m.generator(y), torch.stack(sa), torch.stack(ca)


def load():
    ck = torch.load(os.environ.get('TOY_PT', os.path.join(SCRATCH, 'toy.pt')), map_location='cpu', weights_only=False)
    cfg = Config(**ck['config'])
    m = Transformer(cfg)
    m.load_state_dict(ck['state_dict'])
    m.eval()
    assert ck['vocab'] == task.VOCAB
    return m, cfg, ck


def tables(cfg, n, t, pos=True):
    pe = sinusoid_table(cfg.max_len, cfg.d_model, cfg.pos_layout)
    if not pos:
        pe = torch.zeros_like(pe)
    return pe[:n], pe[:t]


def causal(t, on=True):
    return torch.tril(torch.ones(t, t)) if on else torch.ones(t, t)


@torch.no_grad()
def greedy_wrapped(enc, dec, cfg, src, pos=True, mask=True, max_len=24):
    """Greedy decoding with the two wrappers (batch of sentences of equal length)."""
    b, n = src.shape
    pe_n, _ = tables(cfg, n, 1, pos)
    _, _, memory, _ = enc(src, pe_n)
    out = torch.full((b, 1), task.BOS, dtype=torch.long)
    done = torch.zeros(b, dtype=torch.bool)
    for _ in range(max_len):
        t = out.size(1)
        _, pe_t = tables(cfg, n, t, pos)
        logits, _, _ = dec(out, memory, src, pe_t, causal(t, mask))
        nxt = logits[:, -1].argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, cfg.pad_id), nxt)
        out = torch.cat([out, nxt[:, None]], 1)
        done |= nxt == task.EOS
        if bool(done.all()):
            break
    return out


def exact(dates, outs):
    right, wrong = 0, []
    for d, ids in zip(dates, outs.tolist()):
        p = task.decode_ids(ids)
        if p == task.human(d):
            right += 1
        elif len(wrong) < 12:
            wrong.append([task.iso(d), task.human(d), p])
    return right / len(dates), wrong


def main():
    import onnx
    import onnxruntime as ort
    m, cfg, ck = load()
    enc, dec = EncoderStep(m).eval(), DecoderStep(m).eval()
    n, t = 11, 7
    src = torch.tensor([task.encode_src('2021-03-12')])
    pe_n, pe_t = tables(cfg, n, t)
    tgt = torch.tensor([task.encode_tgt('12 March 2021')[:t]])
    with torch.no_grad():
        _, _, memory, _ = enc(src, pe_n)
    f_enc, f_dec = os.path.join(PUB, 'encoder.onnx'), os.path.join(PUB, 'decoder.onnx')
    kw = dict(opset_version=17, do_constant_folding=True, dynamo=False)
    torch.onnx.export(enc, (src, pe_n), f_enc, input_names=['src_ids', 'pe'],
                      output_names=['emb', 'x0', 'memory', 'enc_att'],
                      dynamic_axes={'src_ids': {1: 'n'}, 'pe': {0: 'n'}, 'emb': {1: 'n'}, 'x0': {1: 'n'},
                                    'memory': {1: 'n'}, 'enc_att': {2: 'n', 3: 'n'}}, **kw)
    torch.onnx.export(dec, (tgt, memory, src, pe_t, causal(t)), f_dec,
                      input_names=['tgt_ids', 'memory', 'src_ids', 'pe', 'mask'],
                      output_names=['logits', 'self_att', 'cross_att'],
                      dynamic_axes={'tgt_ids': {1: 't'}, 'memory': {1: 'n'}, 'src_ids': {1: 'n'}, 'pe': {0: 't'},
                                    'mask': {0: 't', 1: 't'}, 'logits': {1: 't'},
                                    'self_att': {2: 't', 3: 't'}, 'cross_att': {2: 't', 3: 'n'}}, **kw)
    for f in (f_enc, f_dec):
        onnx.checker.check_model(onnx.load(f))
    so = ort.SessionOptions()
    s_enc = ort.InferenceSession(f_enc, so, providers=['CPUExecutionProvider'])
    s_dec = ort.InferenceSession(f_dec, so, providers=['CPUExecutionProvider'])

    train, test = task.split_dates()
    full_pe = sinusoid_table(cfg.max_len, cfg.d_model, cfg.pos_layout).numpy()

    def onnx_greedy(text, max_len=24):
        s = np.array([task.encode_src(text)], dtype=np.int64)
        nn_ = s.shape[1]
        _, _, mem, _ = s_enc.run(None, {'src_ids': s, 'pe': full_pe[:nn_]})
        out = [task.BOS]
        all_logits = []
        for _ in range(max_len):
            tt = len(out)
            lg, _, _ = s_dec.run(None, {'tgt_ids': np.array([out], dtype=np.int64), 'memory': mem, 'src_ids': s,
                                        'pe': full_pe[:tt], 'mask': np.tril(np.ones((tt, tt), np.float32))})
            all_logits.append(lg[0, -1])
            nxt = int(lg[0, -1].argmax())
            out.append(nxt)
            if nxt == task.EOS:
                break
        return out, np.stack(all_logits)

    # 1. teacher-forced logits, onnxruntime vs model.forward, and greedy text vs model.greedy_decode
    sample = test[:300]
    maxdiff, same_text = 0.0, 0
    t0 = time.time()
    for d in sample:
        s = torch.tensor([task.encode_src(task.iso(d))])
        y = torch.tensor([task.encode_tgt(task.human(d))[:-1]])
        with torch.no_grad():
            ref = m(s, y)[0].numpy()
        nn_, tt = s.shape[1], y.shape[1]
        _, _, mem, _ = s_enc.run(None, {'src_ids': s.numpy(), 'pe': full_pe[:nn_]})
        lg, _, _ = s_dec.run(None, {'tgt_ids': y.numpy(), 'memory': mem, 'src_ids': s.numpy(), 'pe': full_pe[:tt],
                                    'mask': np.tril(np.ones((tt, tt), np.float32))})
        maxdiff = max(maxdiff, float(np.abs(lg[0] - ref).max()))
        g = greedy_decode(m, s, task.BOS, task.EOS, max_len=24)[0].tolist()
        o, _ = onnx_greedy(task.iso(d))
        same_text += task.decode_ids(g) == task.decode_ids(o)
    print(f'onnxruntime vs PyTorch, {len(sample)} held-out dates: max |logit diff| {maxdiff:.2e}, '
          f'same greedy text {same_text}/{len(sample)}  ({time.time() - t0:.0f} s)')

    # 2. exact match on all held-out dates: normal, no positions, no mask (PyTorch wrappers; batch by batch)
    abl = {}
    for name, pos, mask in (('normal', True, True), ('no_position', False, True), ('no_mask', True, False),
                            ('neither', False, False)):
        outs = []
        for k in range(0, len(test), 500):
            part = test[k:k + 500]
            srcb = torch.tensor([task.encode_src(task.iso(d)) for d in part])
            outs.append(greedy_wrapped(enc, dec, cfg, srcb, pos, mask))
        width = max(o.size(1) for o in outs)
        outs = torch.cat([torch.nn.functional.pad(o, (0, width - o.size(1)), value=cfg.pad_id) for o in outs])
        acc, wrong = exact(test, outs)
        abl[name] = {'exact': acc, 'right': round(acc * len(test)), 'n': len(test), 'wrong_examples': wrong}
        print(f'  {name:12s} exact match {acc:.4f} on {len(test)} held-out dates; e.g. {wrong[:3]}')
    # the wrapper with the normal table and mask must give exactly what greedy_decode gives
    srcb = torch.tensor([task.encode_src(task.iso(d)) for d in test[:500]])
    a = greedy_wrapped(enc, dec, cfg, srcb)
    b = greedy_decode(m, srcb, task.BOS, task.EOS, max_len=24)
    same_wrapper = bool(torch.equal(a, b[:, :a.size(1)]) and a.size(1) == b.size(1))
    print('wrapper greedy == model.greedy_decode on 500 dates:', same_wrapper)

    # 3. reference runs for the browser check (check_browser.mjs reads this)
    refs = []
    for text in ['2021-03-12', '1969-07-20', '2000-01-01', '1999-12-31', '2045-11-05']:
        ids, lg = onnx_greedy(text)
        s = torch.tensor([task.encode_src(text)])
        with torch.no_grad():
            pt = m(s, torch.tensor([ids[:-1]]))[0].numpy()
        refs.append({'input': text, 'output': task.decode_ids(ids), 'ids': ids, 'logits': pt.round(5).tolist()})
    json.dump(refs, open(os.path.join(RES, 'toy_reference.json'), 'w'))

    log = json.load(open(os.path.join(SCRATCH, 'toy_log.json'))) if os.path.exists(os.path.join(SCRATCH, 'toy_log.json')) else {}
    meta = {
        'task': ck.get('task'), 'config': {k: getattr(cfg, k) for k in ('vocab_size', 'd_model', 'n_heads', 'd_ff',
                'n_encoder_layers', 'n_decoder_layers', 'max_len', 'pad_id', 'pos_layout', 'activation')},
        'vocab': task.VOCAB, 'bos': task.BOS, 'eos': task.EOS, 'pad': task.PAD,
        'params': count_parameters(m), 'first': task.iso(task.FIRST), 'last': task.iso(task.LAST),
        'n_train': len(train), 'n_test': len(test),
        'train': {'steps': log['log'][-1]['step'] if log.get('log') else None,
                  'final_loss': log['log'][-1]['loss'] if log.get('log') else None,
                  'seconds': log.get('seconds')},
        'ablation': abl,
        'check': {'max_logit_diff_onnx_vs_torch': maxdiff, 'same_greedy_text': same_text, 'n': len(sample)},
        'bytes': {'encoder': os.path.getsize(f_enc), 'decoder': os.path.getsize(f_dec)},
        'test_sample': [task.iso(d) for d in test[:40]],
        'train_sample': [task.iso(d) for d in train[:5]],
    }
    json.dump(meta, open(os.path.join(PUB, 'meta.json'), 'w'), separators=(',', ':'))
    json.dump({**meta, 'wrapper_equals_greedy_decode': same_wrapper, 'torch': torch.__version__,
               'onnxruntime': ort.__version__}, open(os.path.join(RES, 'toy_export.json'), 'w'), indent=1)
    print('sizes', meta['bytes'], 'params', meta['params'])


if __name__ == '__main__':
    main()
