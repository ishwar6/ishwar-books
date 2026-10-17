# The ViT demo (/papers/vit/demo): how its files are made

The page is `src/demos/vit/` (React). Every number it shows comes from the files in `public/demos/vit/`, made by the
scripts here, or from the live model running in the reader's browser.

| Script | What it makes |
|---|---|
| `vitwrap.py` | ViT written out step by step (Eq. 1 to 4, Appendix A) on the Hugging Face weights. Same logits as the library (max difference 0.0), plus the patch embeddings, z_0 and every attention map. Input `pos_scale` = 0 leaves out the position embeddings. |
| `pictures.py` | downloads the 8 curated COCO val2017 photos (CC BY 2.0 or CC BY-SA 2.0, Flickr links in the file) and writes `img/<slug>.png` at 224 x 224 (PIL bilinear, like the model's preprocessing). The browser reads these PNGs unchanged. |
| `export_models.py` | exports ViT-B/16, DeiT-S/16 and DeiT-Ti/16 to ONNX (float32, float16 weights, int8), measures every copy against PyTorch and copies the live model to `model/deit-ti16-fp16.onnx`. Writes `results/models.json`. |
| `precompute.py` | the saved ViT-B/16 runs of the curated pictures (`vit-b16/`) and the weights-only files of both models (`meta.json`, `pos.bin`, `possim.bin`). The file format is in its docstring. |
| `check_live.py` | compares the browser's live DeiT-Ti/16 run with float32 PyTorch on the same pictures. Writes `results/check_live.json`. |

```bash
pip install torch transformers pillow onnx onnxruntime
python pictures.py
python export_models.py          # about 10 minutes on a CPU; float32 files go to ~/.cache/vit-demo
python precompute.py             # about 5 minutes
# after a playwright run that saves window.__vitDemo results to live.json:
python check_live.py live.json
```

## Why the browser runs DeiT-Ti/16 and not ViT-B/16

Measured by `export_models.py` (results/models.json), on the 8 curated pictures, against float32 PyTorch:

| Model | Copy | File | Max logit difference | Max attention difference | Same top-5 order |
|---|---|---|---|---|---|
| ViT-B/16 | float16 weights | 173.4 MB | 0.0022 | 0.0030 | 8 of 8 |
| ViT-B/16 | int8 | 89.3 MB | 0.74 | 0.43 | 3 of 8 |
| DeiT-S/16 | float16 weights | 44.3 MB | 0.0066 | 0.018 | 7 of 8 |
| DeiT-S/16 | int8 | 23.5 MB | 1.04 | 0.89 | 2 of 8 |
| DeiT-Ti/16 | float16 weights | 11.6 MB | 0.0055 | 0.0089 | 8 of 8 |

ViT-B/16 is too large for a web page at a precision that keeps its attention maps right, and int8 changes the attention
maps a lot (the demo is about those maps). DeiT-Ti/16 has the same design (16 x 16 patches, 197 tokens, 12 layers,
pre-LayerNorm encoder, [class] token) at D = 192 with 3 heads, and its float16 copy is 11.6 MB with faithful outputs.
So the curated pictures show saved ViT-B/16 numbers, and the live run (any curated picture, or the reader's own photo)
uses DeiT-Ti/16.

The full ViT-B/16 attention maps of a curated picture are 22 MB as float32; `precompute.py` stores each row's patch
weights in 63 levels of the row's largest patch weight, plus the exact [class] weight, gzip-compressed: about 2.2 MB per
picture. The rollout and mean attention distance of the saved runs are computed in Python from the exact maps.
