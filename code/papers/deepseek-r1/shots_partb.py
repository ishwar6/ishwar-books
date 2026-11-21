"""Highlighted excerpts for the Background page (the road to R1): GPT-1 (pre-training loss), InstructGPT (SFT,
reward model, RL with a KL penalty), chain-of-thought, self-consistency, PPO, Let's Verify Step by Step,
reward-model overoptimization, test-time compute, DeepSeekMath (GRPO), DeepSeek-V3 and the R1 paper's own §1."""
from paper_shots import run

GPT1, IGPT, COT, SC, PPO = 'gpt1', '2203.02155', '2201.11903', '2203.11171', '1707.06347'
PRM, GAO, SNELL, DSM, V3 = '2305.20050', '2210.10760', '2408.03314', '2402.03300', '2412.19437'

JOBS = [
    # ---------------------------------------------------------------- pre-training and SFT
    dict(name='pb-gpt1-loss', arxiv_id=GPT1, anchor='Given an unsupervised corpus of tokens',
         highlight=['standard language modeling', 'objective to maximize the following likelihood', 'where k is the size of the context window'],
         above=4, end='These parameters are trained using stochastic gradient descent'),
    dict(name='pb-igpt-steps', arxiv_id=IGPT, page=5, anchor='Step 1: Collect demonstration data, and train a supervised policy.',
         highlight=['Collect demonstration data, and train a supervised policy.', 'Collect comparison data, and train a reward model.',
                    'Optimize a policy against the reward model using PPO.'],
         above=4, end='algorithm (Schulman et al., 2017).'),
    # ---------------------------------------------------------------- chain of thought, self-consistency
    dict(name='pb-cot-abstract', arxiv_id=COT, page=0, anchor='We explore how generating a chain of thought',
         highlight=['a series of intermediate reasoning', 'steps', 'emerge', 'naturally in sufﬁciently large language models'],
         hl_nth={'steps': 0}, above=4, end='surpassing even ﬁnetuned GPT-3 with a veriﬁer.'),
    dict(name='pb-cot-figure1', arxiv_id=COT, page=0, anchor='Figure 1: Chain-of-thought prompting enables large language models',
         highlight=['Chain-of-thought reasoning processes are highlighted.'], figure=True, column='full', below=0),
    dict(name='pb-sc-abstract', arxiv_id=SC, page=0, anchor='Chain-of-thought prompting combined with pre-trained',
         highlight=['samples a diverse set of reasoning paths', 'selects the most consistent answer',
                    'marginalizing out the sampled reasoning paths', 'GSM8K (+17.9%)'],
         above=4, end='ARC-challenge (+3.9%).'),
    # ---------------------------------------------------------------- RL, PPO
    dict(name='pb-ppo-pg', arxiv_id=PPO, page=1, anchor='Policy gradient methods work by computing an estimator',
         highlight=['The most commonly used gradient estimator has the', 'is an estimator of the advantage function at timestep t'],
         above=4, end='is an estimator of the advantage function at timestep t.'),
    dict(name='pb-ppo-clip', arxiv_id=PPO, page=2, anchor='The main objective we propose is the following',
         highlight=['say, ϵ = 0.2', 'removes the incentive for moving rt outside of the', 'interval [1 −ϵ, 1 + ϵ]',
                    'we take the minimum of the clipped and unclipped objective'],
         above=4, end='include it when it makes the objective worse.'),
    # ---------------------------------------------------------------- InstructGPT reward model and RL
    dict(name='pb-igpt-rm', arxiv_id=IGPT, page=7, anchor='Speciﬁcally, the loss function for the reward model is',
         highlight=['is the scalar output of the reward model', 'is the preferred completion out of the pair'],
         above=4, below=86),
    dict(name='pb-igpt-rl', arxiv_id=IGPT, page=8, anchor='Reinforcement learning (RL).',
         highlight=['produces a reward determined by the reward model and ends the episode',
                    'per-token KL penalty from the SFT model at each token',
                    'The value function is initialized from the RM.'],
         above=4, end='models “PPO.”'),
    dict(name='pb-igpt-6b', arxiv_id=IGPT, page=41, anchor='As previously mentioned, for all PPO models we use a 6B RM',
         highlight=['we use a 6B RM and a 6B value function, and the latter is initialized from the former', 'The PPO clip ratio is set to 0.2'],
         above=30, end='policies of all model sizes'),
    dict(name='pb-dsm-critic', arxiv_id=DSM, page=12, anchor='As the value function employed in PPO is typically',
         highlight=['another model of comparable size as', 'the policy model', 'substantial memory and computational burden',
                    'only the last token is assigned a', 'reward score by the reward model'],
         above=4, end='accurate at each token.'),
    # ---------------------------------------------------------------- outcome vs process rewards, reward hacking
    dict(name='pb-prm-abstract', arxiv_id=PRM, page=0, anchor='To train more reliable models',
         highlight=['outcome supervision, which provides', 'feedback for a ﬁnal result', 'process supervision, which provides feedback',
                    'for each intermediate reasoning step', 'process supervision significantly outper', 'solves 78% of'],
         above=30, end='problems from a representative subset of the MATH test set.'),
    dict(name='pb-gao-goodhart', arxiv_id=GAO, page=0, anchor='Goodhart’s law is an adage that states',
         highlight=['When a measure becomes a target, it ceases to be a good', 'measure.',
                    'Optimizing too much against such a model', 'eventually hinders the true objective'],
         hl_nth={'measure.': 0}, above=4, end='a phenomenon we refer to as overoptimization.'),
    # ---------------------------------------------------------------- test-time compute, DeepSeek's pieces
    dict(name='pb-snell-abstract', arxiv_id=SNELL, page=0, anchor='Additionally, in a FLOPs-matched evaluation',
         highlight=['test-time compute can be used to outperform a 14× larger model'], above=4,
         end='test-time compute can be used to outperform a 14× larger model.'),
    dict(name='pb-dsm-abstract', arxiv_id=DSM, page=0, anchor='Second,', highlight=[
        'Group Relative Policy Optimization', '(GRPO), a variant of Proximal Policy Optimization (PPO)', 'optimizing the memory usage of PPO'],
         above=4, end='optimizing the memory usage of PPO.'),
    dict(name='pb-v3-abstract', arxiv_id=V3, page=0, anchor='We present DeepSeek-V3, a strong Mixture-of-Experts',
         highlight=['671B total', 'parameters with 37B activated for each token', '14.8 trillion', 'followed by Supervised Fine-Tuning and Reinforcement Learning stages'],
         above=4, end='fully harness its capabilities.'),
    # ---------------------------------------------------------------- the R1 paper itself, §1
    dict(name='pb-r1-o1', page=2, anchor='In the context of reasoning capabilities',
         highlight=['were the first to introduce inference-time scaling by increasing the length of the Chain-of-',
                    'Thought reasoning process', 'none of these methods has achieved general reasoning'],
         above=4, end='performance comparable to OpenAI’s o1 series models.'),
    dict(name='pb-r1-pure', page=2, anchor='In this paper, we take the first step toward improving',
         highlight=['using pure reinforcement learning (RL)', 'without any supervised data',
                    'DeepSeek-V3-Base as the base model', 'GRPO (Shao et al., 2024) as the RL framework'],
         above=4, end='as the RL framework to improve model performance in reasoning.'),
]

# (left px) to cut: arXiv's rotated side stamp sits left of the text in these crops
TRIM = {'pb-cot-abstract': 300, 'pb-cot-figure1': 215, 'pb-sc-abstract': 300, 'pb-prm-abstract': 340, 'pb-gao-goodhart': 215,
        'pb-snell-abstract': 100, 'pb-dsm-abstract': 115, 'pb-v3-abstract': 122}

if __name__ == '__main__':
    import os
    from PIL import Image
    from paper_shots import OUT
    run(JOBS)
    only = os.environ.get('ONLY')
    for name, left in TRIM.items():
        if only and name not in only.split(','):
            continue
        f = os.path.join(OUT, name + '.png')
        im = Image.open(f)
        im.crop((left, 0, im.width, im.height)).save(f)
