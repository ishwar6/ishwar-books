---
base_model: Qwen/Qwen2.5-0.5B-Instruct
library_name: peft
tags:
- base_model:adapter:Qwen/Qwen2.5-0.5B-Instruct
- lora
- transformers
---
# Chapter 7 reward model (LoRA adapter)

A reward model for Chapter 8: Qwen/Qwen2.5-0.5B-Instruct with a scalar score head, adapted with LoRA
(rank 16, alpha 32, all seven linear layers of every block; the 896 x 1 `score` head trained in full).

The weights file (`adapter_model.safetensors`, 34 MB) is not stored in this repository. Run `ch7_train.py` to recreate it.
Trained by `code/training/ch7_train.py` on 3,200 UltraFeedback binarized pairs, one epoch, Bradley-Terry loss.
Held-out accuracy on 600 UltraFeedback test pairs: 69.5% (95% bootstrap interval 65.7% to 73.2%); see results/ch7_eval_stdout.txt. Trained with learning rate 3e-4 (`python ch7_train.py`), 38 minutes on an Apple M5 Pro.

Only the adapter is stored here (a few tens of MB); the backbone is downloaded from the Hugging Face Hub.

## Loading it

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BACKBONE = 'Qwen/Qwen2.5-0.5B-Instruct'
tok = AutoTokenizer.from_pretrained(BACKBONE)
base = AutoModelForSequenceClassification.from_pretrained(BACKBONE, num_labels=1, dtype=torch.float32)
base.config.pad_token_id = tok.pad_token_id          # needed: the reward is read at the last non-pad token
rm = PeftModel.from_pretrained(base, 'code/training/results/ch7_rm').eval()

def reward(prompt, response):
    msgs = [{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': response}]
    text = tok.apply_chat_template(msgs, tokenize=False).rstrip('\n')   # ends with <|im_end|>, as in training
    ids = tok(text, return_tensors='pt')
    with torch.no_grad():
        return rm(**ids).logits[0, 0].item()
```

Or, from `code/training`: `from ch7_common import load_rm, score_texts; rm = load_rm(); score_texts(rm, prompts, responses)`.

## Things to know before using it as an RL reward

- Format must match training: Qwen chat template with the default system line, final newline stripped, so the
  last token is `<|im_end|>`. Truncate chats to 512 tokens (training never saw longer ones).
- The scale is arbitrary (the loss only sees differences). Normalise before RL, e.g. subtract the mean reward of
  the starting policy's samples and divide by their standard deviation. Raw reward stats on the test answers are in
  results/ch7_eval_stdout.txt.
- It is a weak proxy (0.5B parameters, 3,200 pairs). It likes longer answers and can be gamed (Chapter 7, Sections 7.9
  and 7.10). Use a KL penalty and check outputs with an independent judge.
### Framework versions

- PEFT 0.21.2