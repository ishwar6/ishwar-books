"""Part 8, check 1: our GRPO pieces, on tiny numbers and against TRL.
(a) Eq. 3, the clip of Eq. 1 and the KL of Eq. 2 on hand-sized numbers.
(c) The reward rules and the completion mask on hand-made cases (only a tokenizer is loaded).
(b) The full loss on one fixed batch of real completions of Qwen2.5-0.5B-Instruct: our grpo_loss against the code path
    of TRL's GRPOTrainer (trainer._compute_loss), with the same tensors, on the CPU in float32."""
import math, os, sys, random, statistics, tempfile
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
from minir1.template import make_prompt
from minir1.rewards import accuracy_reward, format_reward, to_number
from minir1.grpo import sample_group, per_token_logps, group_advantages, grpo_loss, completion_lengths
from minir1.train import load_gsm8k, rows

log = common.Log('r1_part8_check')
out = {}

log('== (a) the three equations on small numbers')
r = torch.tensor([2.0, 0.0, 1.0, 0.0])
A = group_advantages(r, 4)
by_hand = [(x - statistics.mean(r.tolist())) / (statistics.stdev(r.tolist()) + 1e-4) for x in r.tolist()]
log('Eq. 3, rewards of one group:', r.tolist())
log('  mean', statistics.mean(r.tolist()), ' std (n - 1)', round(statistics.stdev(r.tolist()), 4))
log('  advantages (ours)      :', [round(a, 4) for a in A.tolist()])
log('  advantages (statistics):', [round(a, 4) for a in by_hand])
log('  all four the same reward:', group_advantages(torch.tensor([1.0, 1.0, 1.0, 1.0]), 4).tolist())
eps = 0.2
for adv, ratio in [(1.0, 1.5), (1.0, 0.7), (-1.0, 0.5), (-1.0, 1.3)]:
    clipped = min(max(ratio, 1 - eps), 1 + eps)
    log(f'Eq. 1 term, A = {adv:+.0f}, ratio {ratio}: min({ratio * adv:+.2f}, {clipped * adv:+.2f}) = '
        f'{min(ratio * adv, clipped * adv):+.2f}')
# Which tokens count as clipped: only those where the min picks the flat, clipped branch.
for adv, ratio in [(1.0, 1.5), (1.0, 0.7), (-1.0, 0.5), (-1.0, 1.3)]:
    _, s1 = grpo_loss(torch.tensor([[math.log(ratio)]]), torch.zeros(1, 1), torch.zeros(1, 1), torch.tensor([adv]),
                      torch.ones(1, 1), eps, 0.0)
    log(f'  clipped? A = {adv:+.0f}, ratio {ratio}: {"yes" if s1["clipped_sum"] else "no"}')
for p_ref, p in [(0.5, 0.5), (0.5, 0.6), (0.5, 0.4), (0.5, 0.9)]:
    x = p_ref / p
    log(f'Eq. 2, pi_ref {p_ref}, pi_theta {p}: {x:.4f} - ln {x:.4f} - 1 = {x - math.log(x) - 1:.5f}')

# A token with A = 0 still feels the KL term: d(loss)/d(log pi_theta) = beta * (1 - pi_ref / pi_theta).
lp = torch.tensor([[math.log(0.6)]], requires_grad=True)
loss0, _ = grpo_loss(lp, lp.detach(), torch.tensor([[math.log(0.5)]]), torch.tensor([0.0]), torch.ones(1, 1), eps, 0.04)
loss0.backward()
log(f'A = 0, pi_theta 0.6, pi_ref 0.5, beta 0.04: gradient of the loss on log pi_theta = {lp.grad.item():.6f}')

log('')
log('== (c) the reward rules and the completion mask on hand-made cases')
for ans, gold in [('18', '18'), ('$1,250.00', '1250'), ('3 + 4 = 7', '7'), ('1/2', '2'), ('1e3', '3'),
                  ('\\boxed{36/2}', '18'), ('10 - 2', '8'), ('2^3', '8')]:
    log(f'accuracy, answer {ans!r:>16} gold {gold:>4}: value {str(to_number(ans)):>5}  reward '
        f'{accuracy_reward("<think>x</think><answer>" + ans + "</answer>", gold):.0f}')
for c in ['<think> 2 + 3 = 5 </think> <answer> 5 </answer>', '<think> </think><answer> </answer>',
          '<think> a </think><think> b </think><answer> 5 </answer>', '<think> a </think><answer> 5 </answer> more',
          '<think> a <answer> 5 </answer>']:
    log(f'format {format_reward(c):.0f}: {c!r}')
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
ends = {tk.convert_tokens_to_ids('<|im_end|>'), tk.eos_token_id, tk.pad_token_id}
body = tk.encode(' 5 </answer>')
demo = [('stopped on </answer>, then 2 padding tokens', body + [tk.pad_token_id] * 2),
        ('stopped on a sampled <|im_end|>, then 1 padding token', tk.encode(' 5') + [tk.convert_tokens_to_ids('<|im_end|>'), tk.pad_token_id]),
        ('hit the length limit', tk.encode(' 5 is the answer'))]
for what, ids in demo:
    n = completion_lengths([ids], tk, ends)[0]
    log(f'mask, {what}: tokens {[tk.decode([t]) for t in ids]} -> {[1] * n + [0] * (len(ids) - n)}')

log('')
log('== (b) our loss against TRL on one fixed batch')
with common.heavy('part8'):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
    from datasets import Dataset
    import trl
    name = 'Qwen/Qwen2.5-0.5B-Instruct'
    tok = AutoTokenizer.from_pretrained(name)
    policy = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)        # CPU, float32
    train, _ = load_gsm8k()
    random.seed(1); torch.manual_seed(1)
    qs = random.sample(train, 2)
    batch = sample_group(policy, tok, [make_prompt(q['question']) for q in qs], 4, 96)
    gold = [q['gold'] for q in qs for _ in range(4)]
    real = torch.tensor([accuracy_reward(t, g) + format_reward(t) for t, g in zip(batch['texts'], gold)])
    log('rewards our rules give these 8 answers:', real.tolist())
    # Fixed rewards, so that every term of the loss is active (real rewards of 8 short answers are often all equal).
    rewards = torch.tensor([2.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 2.0])
    adv = group_advantages(rewards, 4)
    log(f'batch: 2 questions x 4 completions, completion tensor {tuple(batch["completion_ids"].shape)}, '
        f'{int(batch["completion_mask"].sum())} real completion tokens')
    log('fixed rewards used for the check:', rewards.tolist())
    log('advantages:', [round(a, 4) for a in adv.tolist()])
    # Make three different policies, as in training: pi_ref (the start), pi_old (after one step), pi_theta (after two).
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-6, weight_decay=0.0)   # the learning rate of our run
    with torch.no_grad():
        ref_logps = per_token_logps(policy, batch)
    for k in range(2):
        if k == 1:
            with torch.no_grad():
                old_logps = per_token_logps(policy, batch)
        opt.zero_grad()
        lp = per_token_logps(policy, batch)
        loss, _ = grpo_loss(lp, (old_logps if k else ref_logps), ref_logps, adv, batch['completion_mask'])
        loss.backward(); opt.step()
    opt.zero_grad()

    ds = Dataset.from_list([{'prompt': 'x'}] * 8)
    tmp = tempfile.mkdtemp()
    # use_bias_correction_kl=False: R1's Eq. 2 as written. TRL 1.18 defaults to True (DeepSeek-V3.2's KL x ratio).
    for eps, beta, bias in [(0.2, 0.04, False), (0.02, 0.04, False), (0.2, 0.04, True)]:
        cfg = GRPOConfig(output_dir=tmp, loss_type='grpo', epsilon=eps, beta=beta, num_generations=4,
                         use_bias_correction_kl=bias,
                         per_device_train_batch_size=8, max_completion_length=96, temperature=1.0, use_cpu=True,
                         report_to='none', bf16=False, logging_steps=1)
        trainer = GRPOTrainer(model=policy, reward_funcs=[lambda completions, **kw: [0.0] * len(completions)],
                              args=cfg, train_dataset=ds, processing_class=tok)
        trainer.current_gradient_accumulation_steps = 1
        model = trainer.model.train()
        inputs = dict(prompt_ids=batch['prompt_ids'], prompt_mask=batch['prompt_mask'],
                      completion_ids=batch['completion_ids'], completion_mask=batch['completion_mask'],
                      advantages=adv, old_per_token_logps=old_logps, ref_per_token_logps=ref_logps)
        loss_trl = trainer._compute_loss(model, inputs)
        trl_clip = float(trainer._metrics['train']['clip_ratio/region_mean'][-1])
        trl_kl = float(trainer._metrics['train']['kl'][-1])
        loss_trl.backward()
        g_trl = model.lm_head.weight.grad.clone() if model.lm_head.weight.grad is not None else None
        model.zero_grad()
        logps = per_token_logps(model, batch)
        loss_ours, st = grpo_loss(logps, old_logps, ref_logps, adv, batch['completion_mask'], eps, beta)
        loss_ours.backward()
        g_ours = model.lm_head.weight.grad.clone()
        model.zero_grad()
        trl_logps = trainer._get_per_token_logps_and_entropies(
            model, torch.cat([batch['prompt_ids'], batch['completion_ids']], 1),
            torch.cat([batch['prompt_mask'], batch['completion_mask']], 1), batch['completion_ids'].size(1))[0]
        m = batch['completion_mask'].bool()
        d_logp = (trl_logps - logps).abs()[m].max().item()
        d_loss = abs(loss_trl.item() - loss_ours.item())
        d_grad = (g_trl - g_ours).abs().max().item()
        # The same formula on exactly the same numbers: our grpo_loss fed with the log-probs TRL computed itself.
        with torch.no_grad():
            loss_same, _ = grpo_loss(trl_logps, old_logps, ref_logps, adv, batch['completion_mask'], eps, beta)
        d_same = abs(loss_trl.item() - loss_same.item())
        log(f'epsilon {eps}, beta {beta}, TRL use_bias_correction_kl={bias}:')
        log(f'  tokens clipped, ours {st["clipped_sum"] / st["tokens"]:.4f}, TRL clip_ratio/region_mean {trl_clip:.4f}')
        log(f'  mean KL per token, ours {st["kl_sum"] / st["tokens"]:.6f}, TRL kl {trl_kl:.6f}')
        log(f'  loss, ours: {loss_ours.item():+.8f}')
        log(f'  loss, TRL {trl.__version__} GRPOTrainer._compute_loss: {loss_trl.item():+.8f}')
        log(f'  |difference|: loss {d_loss:.2e}, per-token log-probs {d_logp:.2e}, gradient of lm_head {d_grad:.2e}')
        log(f'  relative difference of the two losses: {d_loss / abs(loss_trl.item()):.1e}')
        log(f'  our loss on the log-probs TRL computed: {loss_same.item():+.8f}, |difference| from TRL {d_same:.2e}')
        out[f'eps{eps}_bias{bias}'] = dict(loss_ours=loss_ours.item(), loss_trl=loss_trl.item(), d_loss=d_loss, d_same=d_same, d_logp=d_logp,
                                d_grad=d_grad, clipped=st['clipped_sum'] / st['tokens'], trl_clip=trl_clip, kl=st['kl_sum'] / st['tokens'])
common.save('r1_part8_check', out)
