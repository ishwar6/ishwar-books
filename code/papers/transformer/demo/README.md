# The Transformer demo (/papers/transformer/demo): how its files are made

The page is `src/demos/transformer/` (React, three tabs). Every number it shows comes from the files in
`public/demos/transformer/`, made by the scripts here, or from the toy model running in the reader's browser.

| Script | What it makes |
|---|---|
| `export_toy.py` | exports our from-scratch Transformer (`../scratch/model.py`, trained by `../scratch/train_toy.py` on "2021-03-12" -> "12 March 2021") to `toy/encoder.onnx` and `toy/decoder.onnx` (one decoder call on the whole prefix). The position table and the decoder mask are graph inputs, so the page can switch them off. Checks onnxruntime against PyTorch and measures exact match with no positions and with no mask. Writes `toy/meta.json`, `results/toy_export.json`, `results/toy_reference.json`. |
| `marian_precompute.py` | runs Helsinki-NLP/opus-mt-en-de (74.4 M weights, too heavy for the page) on 9 sentences in Python: greedy translation, top-5 per step, and every attention weight (encoder self, decoder self, cross; all layers and heads), stored as 8-bit levels with a float16 scale per row. Writes `marian/`, `results/marian.json`, `results/marian_reference.json`. |
| `check_browser.mjs` | opens the page with playwright and compares the browser with Python: the toy model's text and logits, and the saved attention rows. Writes `results/browser_check.txt`. |

```bash
pip install --target <folder> onnx onnxruntime onnxscript   # then PYTHONPATH=<folder>; torch and transformers as usual
python export_toy.py            # about 1 minute on a CPU; needs ../scratch/toy.pt
python marian_precompute.py     # downloads the model once (about 300 MB)
node check_browser.mjs http://localhost:4190
```

## Results (this machine, CPU)

- Toy model: 930,944 weights, 2 + 2 layers, d_model 128, 4 heads. ONNX files 1.6 MB + 2.2 MB, float32.
  onnxruntime vs PyTorch on 300 held-out dates: max |logit difference| 7.3e-6, same greedy text 300/300.
  Browser (onnxruntime-web 1.22, wasm) vs PyTorch on 5 dates: same output 5/5, max |logit difference| 6.2e-6.
- Exact match on the 2,000 held-out dates: 100% normal, 0% with no position encoding, 66.8% with no decoder mask.
- Saved translator weights: max |stored - model| 0.0020 over every weight of the 9 sentences; the page shows the same
  values as the float32 model within 0.0019. Encoder layer 5, head 3, "making": difficult 0.71, more 0.29, as Part 6 prints.
