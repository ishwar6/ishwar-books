"""The R1-Zero loop of §2.2: a question batch, G samples each, rule rewards, group advantages, clipped policy
updates with a KL leash to a frozen copy of the starting model. Every step is logged as one line of JSON."""
import copy, gc, json, os, random, time
import psutil
import torch
from .template import make_prompt
from .rewards import accuracy_reward, format_reward
from .grpo import sample_group, per_token_logps, group_advantages, grpo_loss

CFG = dict(model='Qwen/Qwen2.5-0.5B-Instruct', steps=300, minutes=80, questions=4, G=8, max_new_tokens=300,
           temperature=1.0, lr=3e-6, eps=0.2, beta=0.04, minibatches=2, micro=8, seed=0)


def load_gsm8k():
    """GSM8K as (question, gold) pairs; the gold answer is the number after '####'."""
    from datasets import load_dataset
    d = load_dataset('openai/gsm8k', 'main')
    conv = lambda split: [dict(question=x['question'], gold=x['answer'].split('####')[-1].strip().replace(',', ''))
                          for x in split]
    return conv(d['train']), conv(d['test'])


def rows(batch, idx):
    """The completions with indices idx, as a smaller batch with the same keys."""
    return {k: v[idx] for k, v in batch.items() if torch.is_tensor(v)}


def free_memory(device):
    """Give cached memory back after each step. Every step has a different answer length, so the mps allocator keeps
    blocks of many sizes; without this the process grew until the machine swapped."""
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()


def memory_gb():
    """Resident memory of this process in GB (what the operating system sees, including the mps allocations)."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def train(cfg, policy, tok, data, log, jsonl_path, samples_path):
    random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    ref = copy.deepcopy(policy).eval().requires_grad_(False)           # pi_ref: frozen copy of the start
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg['lr'], betas=(0.9, 0.99), weight_decay=0.0)
    G, start, samples = cfg['G'], time.time(), []
    open(jsonl_path, 'w').close()
    log(f'memory at the start: {memory_gb():.1f} GB')
    for step in range(1, cfg['steps'] + 1):
        qs = random.sample(data, cfg['questions'])
        # 1. Sample a group of G answers per question from the current policy (pi_old).
        policy.eval()
        batch = sample_group(policy, tok, [make_prompt(q['question']) for q in qs], G,
                             cfg['max_new_tokens'], cfg['temperature'])
        # 2. Score every answer with the two rules, then compare it with its own group (Eq. 3).
        gold = [q['gold'] for q in qs for _ in range(G)]
        acc = torch.tensor([accuracy_reward(t, g) for t, g in zip(batch['texts'], gold)])
        fmt = torch.tensor([format_reward(t) for t in batch['texts']])
        rewards = acc + fmt                                              # (B,)
        adv = group_advantages(rewards, G).to(policy.device)           # (B,)
        # 3. Log-probs of pi_old and pi_ref, once, before any update of this step.
        B = rewards.numel()
        with torch.no_grad():
            old = torch.cat([per_token_logps(policy, rows(batch, slice(i, i + cfg['micro'])), cfg['temperature'])
                             for i in range(0, B, cfg['micro'])])
            refl = torch.cat([per_token_logps(ref, rows(batch, slice(i, i + cfg['micro'])), cfg['temperature'])
                              for i in range(0, B, cfg['micro'])])
        # 4. Split the answers into mini-batches; one optimiser step each. From the second one on, pi_theta != pi_old.
        policy.train()
        order, stats = torch.randperm(B), []
        for mb in order.chunk(cfg['minibatches']):
            opt.zero_grad()
            for idx in mb.split(cfg['micro']):
                sub = rows(batch, idx.to(policy.device))
                logps = per_token_logps(policy, sub, cfg['temperature'])
                loss, st = grpo_loss(logps, old[idx], refl[idx], adv[idx], sub['completion_mask'],
                                     cfg['eps'], cfg['beta'])
                (loss * len(idx) / len(mb)).backward()                  # mean over the mini-batch
                stats.append(st)                                        # plain floats, no tensors kept
                del sub, logps, loss
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        # 5. Log the step.
        # KL and clipped share: one mean over every real token of the step (each slice has a different token count).
        lengths = batch['completion_mask'].sum(1).float()
        tokens = sum(s['tokens'] for s in stats)
        row = dict(step=step, reward=rewards.mean().item(), acc=acc.mean().item(), fmt=fmt.mean().item(),
                   length=lengths.mean().item(), kl=sum(s['kl_sum'] for s in stats) / tokens,
                   clipped=sum(s['clipped_sum'] for s in stats) / tokens,
                   zero_std_groups=(rewards.view(-1, G).std(1) == 0).float().mean().item(),
                   seconds=round(time.time() - start, 1))
        texts = batch['texts']
        del batch, old, refl, adv
        free_memory(policy.device)
        row['rss_gb'] = round(memory_gb(), 2)
        with open(jsonl_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        log(f"step {step:4d}  reward {row['reward']:.3f}  acc {row['acc']:.3f}  format {row['fmt']:.3f}  "
            f"length {row['length']:6.1f}  kl {row['kl']:.5f}  clipped {row['clipped']:.4f}  "
            f"memory {row['rss_gb']:.1f} GB  {row['seconds']:.0f} s")
        if step in (1, 2, 3) or step % 25 == 0:
            samples.append(dict(step=step, question=qs[0]['question'], gold=qs[0]['gold'],
                                answers=[dict(text=t, acc=a.item(), fmt=f.item())
                                         for t, a, f in zip(texts[:G], acc[:G], fmt[:G])]))
            with open(samples_path, 'w') as f:
                json.dump(samples, f, indent=1)
        if time.time() - start > cfg['minutes'] * 60:
            log(f'time budget of {cfg["minutes"]} minutes reached after step {step}')
            break
    return policy
