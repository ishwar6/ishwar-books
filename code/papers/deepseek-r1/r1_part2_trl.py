"""Method 2 of Part 2: the same GRPO numbers from TRL's own code (trl 1.18, GRPOTrainer).

trl_loss() calls the real function GRPOTrainer._compute_loss. That method normally needs a full trainer
(a model, an accelerator, a dataset). We give it a small stand-in object that has only the settings the method
reads, and a stand-in for the model call that returns our log-probabilities. Every line of the loss itself
(the ratio, the clip, the min, the k3 KL and the masked mean) is TRL's code.
trl_advantages() repeats, line by line, how GRPOTrainer._generate_and_score_completions turns rewards into
advantages (that method also samples text, so it cannot be called on its own)."""
from collections import defaultdict
from types import SimpleNamespace
import torch
from trl import GRPOTrainer
from trl.trainer.utils import nanstd


class _Accel:                           # one process: gathering and reducing change nothing
    def gather(self, x): return x
    def reduce(self, x, reduction='sum'): return x


def trl_loss(logp, logp_old, logp_ref, A, mask, epsilon=0.2, beta=0.04):
    """Same arguments as r1_part2_grpo.grpo_loss. Returns TRL's loss for loss_type='grpo'."""
    me = SimpleNamespace(
        loss_type='grpo', beta=beta, epsilon_low=epsilon, epsilon_high=epsilon,
        importance_sampling_level='token', top_entropy_quantile=1.0, off_policy_mask_threshold=None,
        use_vllm=False, vllm_importance_sampling_correction=False, aux_loss_enabled=False,
        _entropy_bonus_enabled=False, current_gradient_accumulation_steps=1,
        args=SimpleNamespace(use_bias_correction_kl=False, delta=None),
        model=SimpleNamespace(training=True), accelerator=_Accel(),
        _metrics={'train': defaultdict(list), 'eval': defaultdict(list)},
        # the model call: TRL asks for (per-token log-probs, entropies, aux loss); we hand it our log-probs
        _get_per_token_logps_and_entropies=lambda *a, **k: (logp, torch.zeros_like(logp), None),
    )
    B, T = logp.shape
    inputs = {
        'prompt_ids': torch.zeros(B, 1, dtype=torch.long), 'prompt_mask': torch.ones(B, 1, dtype=torch.long),
        'completion_ids': torch.zeros(B, T, dtype=torch.long), 'completion_mask': mask.long(),
        'advantages': A, 'old_per_token_logps': logp_old, 'ref_per_token_logps': logp_ref,
    }
    loss = GRPOTrainer._compute_loss(me, None, inputs)
    return loss, me._metrics['train']


def trl_advantages(rewards, G):
    """The lines of GRPOTrainer._generate_and_score_completions for scale_rewards='group' (TRL's default)."""
    mean_grouped_rewards = torch.nanmean(rewards.view(-1, G), dim=1)
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(G, dim=0)
    std_rewards = nanstd(rewards.view(-1, G), dim=1)
    std_rewards = std_rewards.repeat_interleave(G, dim=0)
    advantages = rewards - mean_grouped_rewards
    advantages = advantages / (std_rewards + 1e-4)
    return advantages
