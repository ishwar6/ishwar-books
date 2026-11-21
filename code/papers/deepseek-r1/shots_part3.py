"""Highlighted excerpts for Part 3 (§2.2.2 reward modeling, §2.2.3 training template, Table 1), plus a few lines of
other papers that the page quotes: Gao et al. (2022) on reward model overoptimization, DeepSeekMath (2024) on outcome
and process supervision, §4.2 of this paper, and the later long version of the R1 paper (Eq. 4 and the maths grader)."""
from paper_shots import run

LONG = '2501.12948'      # the later long version (86 pages), cached next to v1
GAO = '2210.10760'       # Gao, Schulman and Hilton (2022), Scaling laws for reward model overoptimization
DSM = '2402.03300'       # DeepSeekMath (Shao et al., 2024), where GRPO comes from

JOBS = [
    # ---------------------------------------------------------------- v1 page 6 (index 5): Table 1, §2.2.2, §2.2.3
    dict(name='p3-table1', page=5, anchor='A conversation between User and Assistant',
         highlight=['<think> </think>', '<answer> </answer>', 'User: prompt. Assistant:'],
         column='full', above=8, end='during training.'),
    dict(name='p3-reward-source', page=5, anchor='The reward is the source of the training signal',
         highlight=['The reward is the source of the training signal', 'decides the optimization direction of RL',
                    'rule-based reward system'],
         above=18, end='two types of rewards:'),
    dict(name='p3-accuracy', page=5, anchor='Accuracy rewards: The accuracy reward model',
         highlight=['final answer in a specified format (e.g., within a box)', 'reliable',
                    'rule-based verification of correctness', 'a compiler can be used to generate feedback',
                    'predefined test cases'],
         above=4, end='based on predefined test cases.'),
    dict(name='p3-format', page=5, anchor='Format rewards: In addition to the accuracy reward model',
         highlight=['format reward', 'put its thinking process between'], above=4, below=30),
    dict(name='p3-no-neural', page=5, anchor='We do not apply the outcome or process neural reward model',
         highlight=['outcome or process neural reward model', 'reward hacking',
                    'retraining the reward model needs additional training resources',
                    'complicates the whole training pipeline'],
         above=4, end='it complicates the whole training pipeline.'),
    dict(name='p3-template', page=5, anchor='To train DeepSeek-R1-Zero, we begin by designing',
         highlight=['straightforward template', 'first produce a reasoning process, followed by the final answer',
                    'We intentionally limit our constraints to this structural format',
                    'avoiding any content-specific', 'mandating reflective reasoning',
                    'promoting particular problem-solving strategies', 'accurately observe the model’s natural progression'],
         above=18, end='natural progression during the RL process'),
    # ---------------------------------------------------------------- v1 page 15 (index 14): §4.2, the PRM failure
    dict(name='p3-prm-hacking', page=14, anchor='Third, once a model-based PRM is introduced',
         highlight=['it inevitably leads to reward', 'hacking (Gao et al., 2022)'], above=4,
         end='complicates the whole training pipeline.'),
    # ---------------------------------------------------------------- other papers
    dict(name='p3-gao-abstract', arxiv_id=GAO, page=0, anchor='In reinforcement learning from human feedback, it is common',
         highlight=['reward model trained to predict human preferences', 'imperfect proxy',
                    'optimizing its value too much can hinder ground truth performance', 'Goodhart’s law'],
         above=4, end='in accordance with Goodhart’s law.'),
    dict(name='p3-gao-goodhart', arxiv_id=GAO, page=0, anchor='Goodhart’s law is an adage that states',
         highlight=['When a measure becomes a target, it ceases to be a good measure.'], above=4,
         end='a phenomenon we refer to as overoptimization.'),
    dict(name='p3-gao-figure1', arxiv_id=GAO, page=2, anchor='Figure 1: Reward model (RM) parameter size scaling',
         highlight=['the gold reward initially', 'increases and later decreases'], figure=True, column='full', below=0),
    dict(name='p3-dsm-outcome', arxiv_id=DSM, page=13, anchor='4.1.2. Outcome Supervision RL with GRPO',
         highlight=['A reward model is then used to score the outputs'], above=4, end='normalized by subtracting'),
    dict(name='p3-long-eq4', arxiv_id=LONG, page=3, anchor='Format rewards complement the accuracy reward model',
         highlight=['enforcing specific formatting requirements', 'combined with the same weight'],
         above=4, end='combined with the same weight.'),
    dict(name='p3-long-grader', arxiv_id=LONG, page=77, anchor='Parse the final answer within',
         highlight=['use a rule-based grader', 'to parse expressions'], above=30, end='to parse expressions'),
]

if __name__ == '__main__':
    run(JOBS)


def trim():
    """Post-crops: Gao et al. is a one-column paper with arXiv's rotated stamp in the left margin, which the shared
    cropper keeps; cut it off. Figure 1 of Gao et al.: keep panel (b), the RL panel, and the caption."""
    import os
    from PIL import Image
    from paper_shots import OUT
    for name, box in [('p3-gao-abstract', (300, 0, None, None)), ('p3-gao-goodhart', (205, 0, None, None)),
                      ('p3-gao-figure1', (0, 820, None, None))]:
        p = os.path.join(OUT, name + '.png')
        im = Image.open(p)
        if im.info.get('trimmed'):
            continue
        x0, y0, x1, y1 = box
        im = im.crop((x0, y0, x1 or im.width, y1 or im.height))
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo(); meta.add_text('trimmed', '1')
        im.save(p, pnginfo=meta)
        print(f'{name}: trimmed to {im.width}x{im.height}')


if __name__ == '__main__':
    trim()
