"""Highlighted excerpts of the DeepSeek-R1 paper (arXiv v1) for Part 1 (the big idea): title, abstract, Figure 1,
§1, §1.1 and §1.2. Pages are 0-based."""
from paper_shots import run

JOBS = [
    # ---------------------------------------------------------------- page 1: title, abstract, Figure 1
    dict(name='p1-title', page=0, anchor='DeepSeek-R1: Incentivizing Reasoning Capability',
         highlight=['Incentivizing Reasoning Capability', 'via', 'Reinforcement Learning'], column='full', above=8, below=26),
    dict(name='p1-abstract-1', page=0, anchor='We introduce our first-generation reasoning models',
         highlight=['first-generation reasoning models', 'large-scale reinforcement learning (RL) without super-',
                    'vised fine-tuning (SFT) as a preliminary step'],
         above=4, end='remarkable reasoning capabilities.'),
    dict(name='p1-abstract-2', page=0, anchor='Through RL, DeepSeek-R1-Zero naturally emerges',
         highlight=['naturally emerges', 'poor readability, and language', 'mixing'], above=4, end='mixing. To address'),
    dict(name='p1-abstract-3', page=0, anchor='To address these issues and further enhance reasoning performance, we introduce',
         highlight=['multi-stage training and cold-start data before RL', 'comparable to OpenAI-o1-1217'],
         above=4, end='on reasoning tasks. To support'),
    dict(name='p1-abstract-4', page=0, anchor='To support the research community, we open-source',
         highlight=['six dense models', '(1.5B, 7B, 8B, 14B, 32B, 70B) distilled from DeepSeek-R1', 'Qwen and Llama'],
         above=4, end='based on Qwen and Llama.'),
    dict(name='p1-figure1', page=0, anchor='Figure 1 | Benchmark performance of DeepSeek-R1.', figure=True,
         highlight=['Benchmark performance of DeepSeek-R1'], column='full', fig_top=284, below=2),
    # ---------------------------------------------------------------- page 3: §1 Introduction
    dict(name='p1-intro-posttraining', page=2, anchor='Recently, post-training has emerged',
         highlight=['post-training', 'enhance accuracy on reasoning tasks, align with social values', 'adapt',
                    'relatively minimal computational resources against', 'pre-training'],
         above=4, end='against pre-training.'),
    dict(name='p1-intro-o1', page=2, anchor='In the context of reasoning capabilities',
         highlight=['the first to introduce inference-time scaling by increasing the length of the Chain-of-',
                    'Thought reasoning process', 'effective test-time scaling remains an open question'],
         above=4, end='for the research community.'),
    dict(name='p1-intro-prior', page=2, anchor='Several prior works have explored various approaches',
         highlight=['process-based reward models', 'reinforcement learning (Kumar et al., 2024)',
                    'Monte Carlo Tree Search and Beam Search', 'none of these methods has achieved general reasoning',
                    'performance comparable'],
         above=4, end='o1 series models.'),
    dict(name='p1-intro-purerl', page=2, anchor='In this paper, we take the first step',
         highlight=['pure reinforcement learning (RL)', 'without any supervised data',
                    'self-evolution through', 'DeepSeek-V3-Base as the base model', 'GRPO (Shao et al., 2024)'],
         above=4, end='model performance in reasoning.'),
    dict(name='p1-intro-aime', page=2, anchor='During training, DeepSeek-R1-Zero naturally emerged',
         highlight=['thousands of RL steps', 'pass@1 score on AIME 2024 increases from 15.6% to', '71.0%',
                    'with majority voting', '86.7%', 'OpenAI-o1-0912'],
         above=4, end='of OpenAI-o1-0912.'),
    dict(name='p1-intro-r1-1', page=2, anchor='However, DeepSeek-R1-Zero encounters challenges',
         highlight=['poor readability', 'and language', 'mixing', 'small amount of cold-start data',
                    'multi-stage training', 'pipeline'], hl_nth={'pipeline': 1},
         above=4, end='a multi-stage training pipeline.'),
    dict(name='p1-intro-r1-2', page=2, anchor='Specifically, we begin by collecting thousands of cold-start data',
         highlight=['thousands of cold-start data', 'reasoning-oriented RL', 'rejection',
                    'sampling on the RL checkpoint', 'retrain the DeepSeek-V3-Base model', 'additional RL process',
                    'prompts from all scenarios', 'on par with OpenAI-o1-1217'],
         above=4, end='on par with OpenAI-o1-1217.'),
    dict(name='p1-intro-distill', page=2, anchor='We further explore distillation from DeepSeek-R1',
         highlight=['distillation', 'direct distillation from DeepSeek-R1 outperforms applying',
                    'RL on it', 'reasoning patterns discovered by larger base models'],
         above=4, end='improving reasoning capabilities.'),
    dict(name='p1-intro-opensource', page=2, anchor='We open-source the distilled Qwen and Llama',
         highlight=['distilled 14B model outperforms', 'QwQ-32B-Preview', 'new record on the reasoning benchmarks among dense models'],
         above=4, end='among dense models.'),
    # ---------------------------------------------------------------- page 4: §1.1 and §1.2
    dict(name='p1-contrib-zero', page=3, anchor='1.1. Contributions',
         highlight=['directly apply RL to the base model', 'explore chain-of-thought (CoT)',
                    'self-verification, reflection, and generating', 'long CoTs',
                    'incentivized', 'purely through RL, without the need for SFT'],
         above=4, end='advancements in this area.'),
    dict(name='p1-contrib-pipeline', page=3, anchor='We introduce our pipeline to develop DeepSeek-R1',
         highlight=['two RL', 'stages', 'two SFT stages', 'seed'],
         above=4, end='better models.'),
    dict(name='p1-contrib-distill-1', page=3, anchor='Distillation: Smaller Models Can Be Powerful Too',
         highlight=['reasoning patterns of larger models can be distilled into smaller', 'models',
                    'better performance compared to the reasoning patterns discovered', 'through RL on small models'],
         above=4, end='smaller models in the future.'),
    dict(name='p1-contrib-distill-2', page=3, anchor='Using the reasoning data generated by DeepSeek-R1',
         highlight=['55.5% on AIME 2024', '72.6% on AIME 2024, 94.3% on MATH-500', '57.2% on LiveCodeBench',
                    'comparable to o1-mini', '1.5B, 7B, 8B, 14B,', '32B, and 70B'],
         above=4, end='series to the community.'),
    dict(name='p1-results-reasoning', page=3, anchor='1.2. Summary of Evaluation Results',
         highlight=['79.8% Pass@1 on AIME 2024', '97.3%', '2,029 Elo rating on Codeforces', '96.3% human participants',
                    'slightly better than'],
         above=4, end='real world tasks.'),
    dict(name='p1-results-knowledge', page=3, anchor='Knowledge: On benchmarks such as MMLU',
         highlight=['90.8% on MMLU, 84.0% on MMLU-Pro, and 71.5% on GPQA Diamond', 'slightly below that of OpenAI-o1-1217',
                    'SimpleQA'],
         above=4, end='surpasses 4o on this benchmark.'),
    # ---------------------------------------------------------------- page 5: §1.2, last bullet
    dict(name='p1-results-others', page=4, anchor='Others: DeepSeek-R1 also excels',
         highlight=['87.6% on AlpacaEval 2.0', '92.3% on Are-', 'naHard', 'long-context understanding'],
         above=4, end='long-context benchmarks.'),
]

if __name__ == '__main__':
    run(JOBS)
    # page 1: cut away arXiv's rotated side stamp on the left of the abstract excerpts and the figure
    import os
    from PIL import Image
    from paper_shots import OUT
    for j in JOBS:
        if j['page'] == 0 and j['name'] != 'p1-title':
            p = os.path.join(OUT, j['name'] + '.png')
            im = Image.open(p)
            if im.width == 1364:                     # crop only once, on a fresh render
                im.crop((124, 0, im.width, im.height)).save(p)
