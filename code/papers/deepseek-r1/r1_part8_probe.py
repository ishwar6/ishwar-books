"""Part 8, step 0: before training, does the model ever earn a reward? 24 GSM8K training questions, 4 samples each,
the Table 1 template, temperature 1, at most 300 new tokens. Base model (closest to R1-Zero) and Instruct model."""
import os, sys, time, random
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import minir1.template as T
from minir1.template import make_prompt
from minir1.rewards import accuracy_reward, format_reward
from minir1.grpo import sample_group
from minir1.train import load_gsm8k

PREFILL = len(sys.argv) > 1   # run 2: the prompt ends with ' <think>'
log = common.Log('r1_part8_probe' + ('_prefill' if PREFILL else ''))
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
train, _ = load_gsm8k()
random.seed(0)
if not PREFILL:
    T.PREFILL = ''; T.full_completion = lambda g: g
    import minir1.grpo as Gm; Gm.full_completion = T.full_completion
qs = random.sample(train, 24)
res = {}
with common.heavy('part8'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for name in ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B-Instruct']:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
        torch.manual_seed(0)
        t = common.timer(); acc = fmt = n = 0; lens = []; first = None
        for i in range(0, len(qs), 8):
            chunk = qs[i:i + 8]
            b = sample_group(model, tok, [make_prompt(q['question']) for q in chunk], 4, 300)
            lens += b['completion_mask'].sum(1).tolist()
            for j, text in enumerate(b['texts']):
                gold = chunk[j // 4]['gold']
                acc += accuracy_reward(text, gold); fmt += format_reward(text); n += 1
                first = first or text
        log(f'{name}: {n} samples, accuracy reward {acc / n:.3f}, format reward {fmt / n:.3f}, '
            f'mean length {sum(lens) / n:.0f} tokens, {t():.0f} s')
        log('  first sample:', repr(first[:300]))
        res[name] = dict(acc=acc / n, fmt=fmt / n, len=sum(lens) / n)
        del model
common.save('r1_part8_probe' + ('_prefill' if PREFILL else ''), res)
