"""Chapter 5 hands-on: supervised fine-tuning of Qwen2.5-0.5B (base) on 1,000 Alpaca-cleaned examples on the Apple GPU.
Usage: python ch5_sft.py lora   (LoRA r=16 on every linear layer + the embedding rows of <|im_start|> and <|im_end|>, lr 2e-4)
       python ch5_sft.py lora_plain   (LoRA r=16 on every linear layer only: the first run, which could not learn to stop)
       python ch5_sft.py full   (all 494M weights, lr 1e-5, for the forgetting comparison)
Logs the loss curve and the held-out loss, then runs the instruction-following checks and the forgetting probes."""
import sys, time, random, torch
from common import Log, save
from ch4_gpt import lr_at
from ch5_common import load, BASE, load_split, collate, sft_loss, heldout_loss, run_checks, wikitext_ppl, fewshot_antonyms, chat, DEV

MODE = sys.argv[1] if len(sys.argv) > 1 else 'lora'
log = Log(f'ch5_sft_{MODE}')
BS, EPOCHS = 8, 1
LR = 2e-4 if MODE.startswith('lora') else 1e-5

train, val = load_split()
torch.manual_seed(0)
model = load(BASE)                                   # fp32 weights; the matrix multiplications run in bf16 (autocast)
if MODE.startswith('lora'):
    from peft import LoraConfig, get_peft_model
    extra = {} if MODE == 'lora_plain' else {'trainable_token_indices': {'embed_tokens': [151644, 151645]}}   # <|im_start|>, <|im_end|>
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type='CAUSAL_LM',
                     target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'], **extra)
    model = get_peft_model(model, cfg)               # freezes the base weights, adds A and B to each target layer
params = [p for p in model.parameters() if p.requires_grad]
n_train = sum(p.numel() for p in params)
n_all = sum(p.numel() for p in model.parameters())
log(f'mode {MODE}: training {n_train:,} of {n_all:,} parameters ({n_train / n_all:.2%}), lr {LR}, batch {BS}')
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

steps = EPOCHS * len(train) // BS
warmup = max(1, steps // 20)
hist = {'step': [], 'loss': [], 'val_step': [], 'val': [], 'lr': []}
v = heldout_loss(model, val)
hist['val_step'].append(0); hist['val'].append(v)
log(f'step    0  held-out loss {v:.4f}')
order = list(range(len(train)))
random.Random(1).shuffle(order)
t0 = time.time()
model.train()
for step in range(steps):
    lr = lr_at(step, LR, warmup, steps, floor=0.0)
    for g in opt.param_groups:
        g['lr'] = lr
    ids, lab, att = (t.to(DEV) for t in collate([train[i][1] for i in order[step * BS:(step + 1) * BS]]))
    with torch.autocast('mps', dtype=torch.bfloat16):
        loss = sft_loss(model, ids, lab, att)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    hist['step'].append(step + 1); hist['loss'].append(loss.item()); hist['lr'].append(lr)
    if (step + 1) % 25 == 0 or step + 1 == steps:
        msg = f'step {step + 1:4d}  train loss {loss.item():.4f}  lr {lr:.2e}  ({time.time() - t0:.0f}s)'
        if (step + 1) % 50 == 0 or step + 1 == steps:
            v = heldout_loss(model, val)
            hist['val_step'].append(step + 1); hist['val'].append(v)
            msg += f'  held-out loss {v:.4f}'
            model.train()
        log(msg)
hist['seconds'] = time.time() - t0
hist['n_train'], hist['n_all'] = n_train, n_all
log(f'training took {hist["seconds"] / 60:.1f} minutes')

model.eval()
log('instruction-following checks (greedy, 150 new tokens at most):')
hist['checks'] = run_checks(model, log)
hist['pass'] = sum(r['passed'] for r in hist['checks']); hist['stops'] = sum(r['stopped'] for r in hist['checks'])
log(f'passed {hist["pass"]}/{len(hist["checks"])}, ended its turn {hist["stops"]}/{len(hist["checks"])}')
hist['wikitext_ppl'] = wikitext_ppl(model)
hist['antonyms_4shot'] = fewshot_antonyms(model)
log(f'forgetting probes: wikitext-2 perplexity {hist["wikitext_ppl"]:.2f}, 4-shot antonyms {hist["antonyms_4shot"]:.0%}')
hist['samples'] = {}
for q in ["Give three tips for a good night's sleep.", 'Write a short, polite email asking a colleague to send the meeting notes.',
          'What is the difference between weather and climate?']:
    t, s = chat(model, q, 200)
    hist['samples'][q] = {'reply': t, 'stopped': s}
    log(f'Q: {q}\nA: {t[:400]}\n   (ended its turn: {s})')
if MODE.startswith('lora'):
    model.save_pretrained(f'/Users/admin/.cache/ch5_{MODE}_adapter')
save(f'ch5_sft_{MODE}', hist)
