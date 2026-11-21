"""Highlighted excerpts for Part 8 (train a reasoner from scratch): every piece of §2.2 that our minir1 package turns
into code (Table 1, the two rewards, Eq. 1 to 3), the results we compare with (Figure 3, §4.1), and two later sources
for the values v1 does not give (DeepSeekMath's per-token objective, the long version's hyperparameters)."""
from paper_shots import run

LONG = '2501.12948'          # the later 86-page version of the paper (hyperparameters)
MATH = '2402.03300'          # Shao et al. (2024), DeepSeekMath, where GRPO comes from

JOBS = [
    # ------------------------------------------------------------ page 5 (index 4): §2.2 and §2.2.1
    dict(name='p8-pure-rl', page=4, anchor='Reinforcement learning has demonstrated significant effectiveness',
         highlight=['develop reasoning capabilities without any supervised data',
                    'self-evolution through a pure reinforcement learning process'],
         above=4, end='valuable insights.'),
    dict(name='p8-grpo-intro', page=4, anchor='In order to save the training costs of RL',
         highlight=['foregoes the critic model', 'estimates the baseline from group scores instead',
                    'samples a group of outputs'],
         above=4, end='then optimizes the policy model'),
    dict(name='p8-eq1', page=4, anchor='then optimizes the policy model', highlight=[], band=(2, 63), above=4, below=60),
    dict(name='p8-eq2', page=4, anchor='then optimizes the policy model', highlight=[], band=(72, 96), above=-70, below=98),
    dict(name='p8-eq3', page=4, anchor='are hyper-parameters, and', highlight=['computed using a group of rewards'],
         band=(14, 44), above=4, below=46),
    # ------------------------------------------------------------ page 6 (index 5): Table 1, §2.2.2, §2.2.3
    dict(name='p8-table1', page=5, anchor='Table 1 | Template for DeepSeek-R1-Zero', figure=True,
         highlight=['<think> </think>', '<answer> </answer>', 'User: prompt. Assistant:']),
    dict(name='p8-limit', page=5, anchor='To train DeepSeek-R1-Zero, we begin by designing',
         highlight=['limit our constraints to this structural format',
                    'accurately observe the model’s natural progression during the RL'],
         above=16, below=90),
    dict(name='p8-accuracy', page=5, anchor='Accuracy rewards:',
         highlight=['evaluates whether the response is correct', 'provide the final answer in a specified format',
                    'rule-based verification of correctness'],
         above=4, end='predefined test cases.'),
    dict(name='p8-format', page=5, anchor='Format rewards:',
         highlight=['put its thinking process between'], above=4, below=30),
    dict(name='p8-no-neural', page=5, anchor='We do not apply the outcome or process neural reward model',
         highlight=['may suffer from reward hacking'], above=4, end='complicates the whole training pipeline.'),
    # ------------------------------------------------------------ pages 8 and 15: what R1-Zero did, and the small-model warning
    dict(name='p8-fig3', page=7, anchor='Figure 3 | The average response length', figure=True,
         highlight=['naturally learns to solve reasoning tasks with more thinking time'], below=0),
    dict(name='p8-thinking', page=7, anchor='ment throughout the training process',
         highlight=['hundreds to thousands of reasoning tokens'], above=4, end='in greater depth.'),
    dict(name='p8-small', page=14, anchor='Therefore, we can draw two conclusions',
         highlight=['require enormous computational power', 'may not even achieve the performance',
                    'more powerful base models and larger'],
         above=4, end='reinforcement learning.'),
    # ------------------------------------------------------------ other sources
    dict(name='p8-math-eq3', arxiv_id=MATH, page=12, anchor='by maximizing the following objective',
         highlight=[], band=(2, 50), above=4, below=52),
    dict(name='p8-math-outcome', arxiv_id=MATH, page=13, anchor='Outcome Supervision RL with GRPO',
         highlight=['sets the advantages', 'tokens in the output as the normalized reward'],
         above=4, end='objective defined in equation (3).'),
    dict(name='p8-long-hparams', arxiv_id=LONG, page=2, anchor='To train DeepSeek-R1-Zero, we set the learning rate',
         highlight=['learning rate to 3e-6, the KL coefficient to 0.001', 'sampling temperature to 1',
                    'we sample 16 outputs', '32,768 tokens before the 8.2k step and 65,536 tokens afterward',
                    'Every 400 steps, we replace the reference model', 'randomly split into 16 mini-batches'],
         above=4, end='single inner epoch.'),
]

if __name__ == '__main__':
    run(JOBS)
