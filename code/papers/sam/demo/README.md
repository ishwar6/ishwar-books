# The SAM demo (/papers/sam/demo): how its files are made

The page runs the prompt encoder and the mask decoder of the released SAM ViT-H (`facebook/sam-vit-huge`) in the
browser with onnxruntime-web (WebAssembly, one thread). The image encoder is too heavy for a browser, so its output
for each photo is computed here, once, and saved. The page code is in `src/demos/sam/`.

| File | What it does | Writes |
|---|---|---|
| `embed_images.py` | the 8 photos and their image embeddings (256 × 64 × 64, saved as float16) | `public/demos/sam/img/*.jpg`, `emb/*.bin`, `photos.json`, `results/embed_stdout.txt` |
| `export_decoder.py` | one ONNX graph: prompt encoder + mask decoder (4 mask tokens, 4 predicted IoUs, optional mask prompt); checks it against `SamModel` in PyTorch and, if installed, with onnxruntime | `public/demos/sam/sam_decoder.onnx`, `results/export_stdout.txt` |
| `reference.py` | Python masks for 8 prompt sets on 3 photos, and the effect of the float16 embedding | `results/reference.json`, `results/reference_stdout.txt` |
| `check_browser.mjs` | replays the same prompts in the demo page (playwright) and compares each mask with Python | `results/browser_check.txt` |

```bash
cd code/papers/sam/demo
pip install torch transformers datasets pillow onnx onnxruntime
# the COCO licence of each photo: annotations/instances_val2017.json from the COCO 2017 annotations
COCO_INSTANCES_VAL2017=/path/to/instances_val2017.json python embed_images.py
python export_decoder.py
python reference.py
cd ../../../.. && npx vite --port 4190 &        # then, with playwright installed in ~/.venvs/shots:
node code/papers/sam/demo/check_browser.mjs http://localhost:4190
```

The fp32 embeddings for the checks are kept in `~/.cache/papers/sam_demo/`. The encoder cost numbers on the page
(5.96 TFLOPs; 2,999 ms on the laptop GPU, 6,779 ms on the CPU) come from `../results/part3_stdout.txt`.

Photos: COCO val2017 images 39769, 174482, 357888, 364322, 414795, 498919, 530052 (Flickr; the licence of each
is in `photos.json` and on the page) and the car photo of the Hugging Face SAM example files
(`ybelkada/segment-anything`, `assets/car.png`), resized to 1024 × 683.
