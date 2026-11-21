"""Part 8, Method 2: the same recipe with TRL's GRPOTrainer, for 4 steps only (to show the library route, not a run).
Same template, same two reward rules, same GSM8K questions, G = 8, epsilon 0.2, beta 0.04, learning rate 3e-6.
Generation stops at '</answer>' as in our package. One difference is left: after a stop on '</answer>', TRL's mask
also keeps the first padding token (for Qwen it is the end-of-text token, which TRL treats as a sampled end)."""
import functools, os, sys, tempfile
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from minir1.template import make_prompt, full_completion
from minir1.rewards import accuracy_reward, format_reward
from minir1.train import load_gsm8k

log = common.Log('r1_part8_trl')


def accuracy(completions, gold, **kw):               # TRL passes the dataset columns (here: gold) as keyword arguments
    return [accuracy_reward(full_completion(c), g) for c, g in zip(completions, gold)]   # put the prefilled tag back


def fmt(completions, **kw):
    return [format_reward(full_completion(c)) for c in completions]


with common.heavy('part8'):
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    name = 'Qwen/Qwen2.5-0.5B-Instruct'
    train, _ = load_gsm8k()
    ds = Dataset.from_list([dict(prompt=make_prompt(q['question']), gold=q['gold']) for q in train[:64]])
    args = GRPOConfig(
        output_dir=tempfile.mkdtemp(), max_steps=4, learning_rate=3e-6, lr_scheduler_type='constant', loss_type='grpo', beta=0.04, epsilon=0.2,
        use_bias_correction_kl=False,                           # the KL of R1's Eq. 2, as in our package
        num_generations=8, per_device_train_batch_size=16,      # 16 completions per step = 2 questions x G = 8
        max_completion_length=300, temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
        generation_kwargs=dict(stop_strings=['</answer>']),      # stop where our sampler stops
        logging_steps=1, report_to='none',
        bf16=False, save_strategy='no', seed=0)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    model.generate = functools.partial(model.generate, tokenizer=tok)   # generate() needs the tokenizer for a stop string
    trainer = GRPOTrainer(model=model, reward_funcs=[accuracy, fmt], args=args, train_dataset=ds,
                          processing_class=tok)
    t = common.timer()
    trainer.train()
    keys = ['reward', 'rewards/accuracy/mean', 'rewards/fmt/mean', 'completions/mean_length', 'kl',
            'clip_ratio/region_mean', 'frac_reward_zero_std']
    for row in trainer.state.log_history:
        if 'reward' in row:
            log(f"step {row['step']}: " + '  '.join(f'{k} {row[k]:.4g}' for k in keys if k in row))
    log(f'4 steps in {t():.0f} s on {trainer.args.device}')
