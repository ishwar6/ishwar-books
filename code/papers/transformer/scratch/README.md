# The Transformer, from scratch

The model of "Attention Is All You Need" (Vaswani et al., 2017) in readable PyTorch, with one small class per block
of Figure 1. It does not use `nn.MultiheadAttention` or `nn.Transformer`. The page that walks through it:
https://ishwarj.com/papers/transformer/part-7-build-it-from-scratch

| File | What it is |
|---|---|
| `model.py` | The whole model: `TokenEmbedding`, `PositionalEncoding`, `scaled_dot_product_attention`, `MultiHeadAttention`, `FeedForward`, `AddAndNorm`, `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, `Transformer`, the masks, `greedy_decode` and `beam_search`. |
| `load_marian.py` | Copies the weights of `Helsinki-NLP/opus-mt-en-de` into `model.py` and translates. |
| `toy_task.py` | The toy task: `"2021-03-12"` to `"12 March 2021"`, with a 41-token character vocabulary. |
| `train_toy.py` | Trains a small model (930,944 parameters) on the toy task on a laptop CPU in about 4 minutes. |
| `toy.pt`, `toy_log.json` | The trained weights (with config and vocabulary) and the training log. |

## Run it

You need Python 3.10+, `torch` and, for the Marian part, `transformers` and `sentencepiece`.

```bash
cd code/papers/transformer/scratch
python train_toy.py      # trains the date model, writes toy.pt and toy_log.json (about 4 minutes on CPU)
python load_marian.py    # downloads opus-mt-en-de (about 300 MB once), copies the weights, translates two sentences
```

The checks that compare each block with PyTorch and the whole model with `transformers` live one folder up:

```bash
cd code/papers/transformer
python tx_part7.py         # each block vs torch.nn (MultiheadAttention, TransformerEncoderLayer, ...), parameter counts
python tx_part7_marian.py  # our model with Marian's weights vs MarianMTModel: encoder output, logits, translations
python tx_part7_toy.py     # what the date model learned: outputs, failures, cross-attention
```

## Use the trained date model

```python
import torch
from model import Config, Transformer, greedy_decode
from toy_task import encode_src, decode_ids, BOS, EOS

ck = torch.load('toy.pt')
model = Transformer(Config(**ck['config']))
model.load_state_dict(ck['state_dict'])
out = greedy_decode(model, torch.tensor([encode_src('2021-03-12')]), BOS, EOS, max_len=24)
print(decode_ids(out[0].tolist()))   # 12 March 2021
```

`model.encode(src_ids)` returns `(memory, src_mask)` and `model.decode(tgt_ids, memory, src_mask)` returns the logits,
so the encoder and one decoder step can be exported to ONNX one by one. No `forward` has a Python `if` on tensor values.
