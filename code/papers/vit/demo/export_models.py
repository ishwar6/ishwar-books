"""Export ViT models to ONNX for the browser demo, make smaller copies, and measure what each copy costs.

For each model (ViT-B/16 = google/vit-base-patch16-224, the model of the series; DeiT-S/16 and DeiT-Ti/16, smaller
ViTs with the same design) this script
  1. exports vitwrap.ViTSteps (pixels, pos_scale -> patch_embeddings, tokens, attentions, logits) to ONNX, float32;
  2. makes an int8 copy (dynamic quantisation of the weight matrices, onnxruntime.quantization) and a float16 copy
     (weights stored as float16 and turned back into float32 when the model loads, so the browser computes in float32);
  3. runs every copy with onnxruntime on the curated pictures and compares it with the float32 PyTorch model:
     max |logit difference|, same top-1, same top-5, max |attention difference|, time per picture on this CPU.
Writes results/models.json and copies the chosen file to public/demos/vit/model/.
Needs: pip install onnx onnxruntime (on top of torch and transformers).  Run pictures.py first."""
import json, os, shutil, sys, time
import numpy as np
import torch
from PIL import Image
import onnx
from onnx import numpy_helper, helper, TensorProto
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from vitwrap import ViTSteps
from pictures import PICTURES

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'vit')
BUILD = os.environ.get('VIT_DEMO_BUILD', os.path.expanduser('~/.cache/vit-demo'))   # large files, not committed
RES = os.path.join(HERE, 'results')
MODELS = {'vit-b16': 'google/vit-base-patch16-224', 'deit-s16': 'facebook/deit-small-patch16-224',
          'deit-ti16': 'facebook/deit-tiny-patch16-224'}
LIVE = 'deit-ti16'         # the model the browser runs (see the README for why)
LIVE_VARIANT = 'fp16'
os.makedirs(BUILD, exist_ok=True)
os.makedirs(RES, exist_ok=True)


def picture(slug):
    a = np.asarray(Image.open(os.path.join(PUB, 'img', f'{slug}.png')).convert('RGB'), dtype=np.float32)
    return a.transpose(2, 0, 1)[None].copy()                              # (1, 3, 224, 224), 0 to 255


def export(key):
    path = os.path.join(BUILD, f'{key}-fp32.onnx')
    m = ViTSteps(MODELS[key]).eval()
    if not os.path.exists(path):
        x, s = torch.from_numpy(picture('cats')), torch.ones(1)
        torch.onnx.export(m, (x, s), path, input_names=['pixels', 'pos_scale'],
                          output_names=['patch_embeddings', 'tokens', 'attentions', 'logits'],
                          opset_version=17, dynamo=False, do_constant_folding=True)
        onnx.save(onnx.load(path), path)                                   # one file, weights inside
    return m, path


def int8(src, dst):
    if not os.path.exists(dst):
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=['MatMul'],
                         extra_options={'MatMulConstBOnly': True})        # only weight matrices, not q k^T or A v
    return dst


def fp16_weights(src, dst):
    """Store every large float32 weight as float16 and add a Cast back to float32 in front of its users."""
    if os.path.exists(dst):
        return dst
    g = onnx.load(src)
    new_inits, casts = [], []
    for t in g.graph.initializer:
        a = numpy_helper.to_array(t)
        if a.dtype == np.float32 and a.size >= 1024:
            h = numpy_helper.from_array(a.astype(np.float16), t.name + '_fp16')
            new_inits.append(h)
            casts.append(helper.make_node('Cast', [h.name], [t.name], to=TensorProto.FLOAT, name=t.name + '_cast'))
        else:
            new_inits.append(t)
    del g.graph.initializer[:]
    g.graph.initializer.extend(new_inits)
    nodes = casts + list(g.graph.node)
    del g.graph.node[:]
    g.graph.node.extend(nodes)
    onnx.save(g, dst)
    return dst


def compare(m, path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1                                            # like the browser: one thread
    sess = ort.InferenceSession(path, so, providers=['CPUExecutionProvider'])
    rows, times = [], []
    for slug, *_ in PICTURES:
        x = picture(slug)
        with torch.no_grad():
            _, _, at, lg = m(torch.from_numpy(x), torch.ones(1))
        t0 = time.time()
        o = sess.run(['attentions', 'logits'], {'pixels': x, 'pos_scale': np.ones(1, np.float32)})
        times.append(time.time() - t0)
        a, b = lg[0].numpy(), o[1][0]
        rows.append(dict(picture=slug, logit_max_abs_diff=float(np.abs(a - b).max()),
                         same_top1=bool(a.argmax() == b.argmax()),
                         same_top5=bool(list(np.argsort(-a)[:5]) == list(np.argsort(-b)[:5])),
                         top1_prob_torch=float(torch.softmax(torch.from_numpy(a), -1).max()),
                         top1_prob_onnx=float(torch.softmax(torch.from_numpy(b), -1).max()),
                         attention_max_abs_diff=float(np.abs(at.numpy() - o[0]).max())))
    return rows, float(np.median(times))


if __name__ == '__main__':
    keys = sys.argv[1:] or list(MODELS)
    out = json.load(open(os.path.join(RES, 'models.json'))) if os.path.exists(os.path.join(RES, 'models.json')) else {}
    for key in keys:
        m, fp32 = export(key)
        with torch.no_grad():                                              # the step-by-step model equals the library
            x = torch.from_numpy(picture('cats'))
            ref = m.m(pixel_values=(x - m.mean) / m.std).logits
            mine = m(x, torch.ones(1))[3]
        R = dict(name=MODELS[key], params=sum(p.numel() for p in m.m.parameters()), D=m.D, heads=m.H, layers=m.L,
                 steps_vs_library_max_abs_diff=float((ref - mine).abs().max()), variants={})
        print(f'{key}: {R["params"]:,} weights, D = {m.D}, {m.H} heads; step-by-step vs library: {R["steps_vs_library_max_abs_diff"]:.2e}')
        for var, path in [('fp32', fp32), ('fp16', fp16_weights(fp32, fp32.replace('fp32', 'fp16'))),
                          ('int8', int8(fp32, fp32.replace('fp32', 'int8')))]:
            rows, t = compare(m, path)
            mb = os.path.getsize(path) / 1e6
            R['variants'][var] = dict(file_mb=round(mb, 1), ort_cpu_1thread_s=round(t, 3), pictures=rows,
                                      max_logit_diff=max(r['logit_max_abs_diff'] for r in rows),
                                      max_attention_diff=max(r['attention_max_abs_diff'] for r in rows),
                                      top1_agree=sum(r['same_top1'] for r in rows), top5_agree=sum(r['same_top5'] for r in rows),
                                      n=len(rows))
            v = R['variants'][var]
            print(f'  {var}: {mb:6.1f} MB  {t:.2f} s/picture (1 thread)  max |logit diff| {v["max_logit_diff"]:.4f}'
                  f'  max |attention diff| {v["max_attention_diff"]:.4f}  top-1 {v["top1_agree"]}/{v["n"]}  top-5 order {v["top5_agree"]}/{v["n"]}')
        out[key] = R
    json.dump(out, open(os.path.join(RES, 'models.json'), 'w'), indent=1)
    if LIVE in out:
        os.makedirs(os.path.join(PUB, 'model'), exist_ok=True)
        shutil.copy(os.path.join(BUILD, f'{LIVE}-{LIVE_VARIANT}.onnx'), os.path.join(PUB, 'model', f'{LIVE}-{LIVE_VARIANT}.onnx'))
