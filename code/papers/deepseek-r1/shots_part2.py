"""Highlighted excerpts for Part 2 (GRPO): §2.1, §2.2, §2.2.1 and Equations 1 to 3 of DeepSeek-R1 (v1),
the PPO clip of Schulman et al. (2017), GRPO's origin in DeepSeekMath (Shao et al., 2024, §4.1, Figure 4),
and the GRPO settings and the GRPO-versus-PPO notes of the later, long version of the R1 paper."""
from paper_shots import run

DSM = '2402.03300'     # DeepSeekMath
PPO = '1707.06347'     # Proximal Policy Optimization
LONG = '2501.12948'    # the later long version of the R1 paper (86 pages, with the Nature supplement)

JOBS = [
    # ------------------------------------------------------------ R1 v1, page 5 (index 4): §2.1, §2.2, §2.2.1
    dict(name='p2-overview', page=4, anchor='2.1. Overview',
         highlight=['heavily relied on large amounts of supervised data', 'even without using supervised',
                    'small amount of cold-start data', 'applies RL directly to the base model without any SFT data',
                    'Distill the reasoning capability'],
         above=4, end='small dense models.'),
    dict(name='p2-pure-rl', page=4, anchor='2.2. DeepSeek-R1-Zero: Reinforcement Learning on the Base Model',
         highlight=['heavily depended on supervised data, which are time-intensive to gather',
                    'without any supervised data', 'pure reinforcement learning process'],
         above=4, end='valuable insights.'),
    dict(name='p2-grpo-intro', page=4, anchor='2.2.1. Reinforcement Learning Algorithm',
         highlight=['save the training costs of RL', 'foregoes the critic model',
                    'typically the same size as the policy model', 'estimates the baseline from group scores instead',
                    'samples a group of outputs'],
         above=4, end='maximizing the following objective:'),
    dict(name='p2-eq1', page=4, anchor='maximizing the following objective:', band=(4, 50),
         highlight=[], above=2, below=54),
    dict(name='p2-eq2', page=4, anchor='maximizing the following objective:', band=(59, 92),
         highlight=[], above=-70, below=93, snap=False),
    dict(name='p2-eq3', page=4, anchor='are hyper-parameters, and', band=(20, 56),
         highlight=['are hyper-parameters', 'computed using a group of rewards'], above=4, below=62),
    # ------------------------------------------------------------ PPO (Schulman et al., 2017), page 3: the clip
    dict(name='p2-ppo-clip', arxiv_id=PPO, page=2, anchor='The main objective we propose is the following:',
         highlight=['removes the incentive for moving rt outside of the interval', 'pessimistic bound'],
         above=4, end='plots a single term'),
    # ------------------------------------------------------------ DeepSeekMath, §4.1
    dict(name='p2-dsm-ppo', arxiv_id=DSM, page=10, anchor='4.1.1. From PPO to GRPO',
         highlight=['actor-critic RL algorithm', 'Generalized Advantage Estimation (GAE)'],
         above=4, end='(Schulman et al., 2015), based'),
    dict(name='p2-dsm-figure4', arxiv_id=DSM, page=12, anchor='Figure 4 | Demonstration of PPO and our GRPO.',
         highlight=['GRPO foregoes the value model', 'estimating the baseline from group scores'],
         figure=True, column='full', below=26),
    dict(name='p2-dsm-burden', arxiv_id=DSM, page=12, anchor='As the value function employed in PPO',
         highlight=['another model of comparable size as', 'substantial memory and computational burden',
                    'only the last token is assigned a reward score',
                    'uses the average reward of multiple sampled outputs'],
         above=4, end='by maximizing the following objective:'),
    dict(name='p2-dsm-eq3', arxiv_id=DSM, page=12, anchor='by maximizing the following objective:', band=(4, 50),
         highlight=['instead of adding KL', 'directly adding the KL divergence'], above=2,
         end='avoiding complicating the calculation'),
    dict(name='p2-dsm-kl', arxiv_id=DSM, page=13, anchor='And different from the KL penalty term used in',
         highlight=['unbiased estimator (Schulman, 2020)', 'guaranteed to be positive'], above=4,
         end='which is guaranteed to be positive.'),
    dict(name='p2-dsm-outcome', arxiv_id=DSM, page=13, anchor='4.1.2. Outcome Supervision RL with GRPO',
         highlight=['subtracting the group average and dividing by the group standard deviation',
                    'sets the advantages'], above=4, end='maximizing the objective defined in equation (3).'),
    dict(name='p2-dsm-settings', arxiv_id=DSM, page=14, anchor='For GRPO, we set the learning rate',
         highlight=['The KL coefficient is 0.04', 'we sample 64 outputs', 'single update following each exploration stage'],
         above=4, end='following each exploration stage.'),
    # ------------------------------------------------------------ the later long version of the R1 paper
    dict(name='p2-long-settings', arxiv_id=LONG, page=2, anchor='To train DeepSeek-R1-Zero,',
         highlight=['learning rate to 3e-6', 'the KL coefficient to 0.001', 'sample 16 outputs', 'replace the', 'reference model with the latest policy model',
                    'single inner epoch'],
         above=4, end='trained for only a single inner epoch.'),
    dict(name='p2-long-clip', arxiv_id=LONG, page=6, anchor='3.2.1. Training Details of the First RL Stage',
         highlight=['GRPO clip', 'ratio 𝜀to 10'], above=4, end='with a maximum length of 32,768.'),
    dict(name='p2-long-clip-note', arxiv_id=LONG, page=7, anchor='Note that the clip ratio plays a crucial role',
         highlight=['truncation', 'instability during training'], above=4, end='instability during training.'),
    dict(name='p2-long-value', arxiv_id=LONG, page=14, anchor='In contrast, in PPO, the advantage is typically',
         highlight=['inherently', 'difficult, especially when only the final outcome reward is available',
                    'reflection and revision'],
         above=4, end='based on a partial response.'),
]

if __name__ == '__main__':
    run(JOBS)
