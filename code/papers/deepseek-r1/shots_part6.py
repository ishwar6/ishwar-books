"""Highlighted excerpts for Part 6 (distillation): §1, §1.1, §2.3.3, §2.4, §3, Table 5, §4.1, Table 6, §5,
the hyper-parameters of distillation from the later long version (arXiv 2501.12948, Appendix B.4.3),
and two earlier papers: Hinton et al. (2015) and Kim and Rush (2016)."""
from paper_shots import run

LONG = '2501.12948'        # the later 86-page version (cached as ~/.cache/papers/2501.12948.pdf)
HINTON = '1503.02531'      # Distilling the Knowledge in a Neural Network
KIMRUSH = '1606.07947'     # Sequence-Level Knowledge Distillation

JOBS = [
    # ---------------------------------------------------------------- §1 Introduction, page 3 (index 2)
    dict(name='p6-intro', page=2, anchor='We further explore distillation from DeepSeek-R1',
         highlight=['direct distillation from DeepSeek-R1 outperforms applying RL on it',
                    'reasoning patterns discovered by larger base models are cru',
                    'distilled 14B model outperforms state-of-the-art open-source'],
         above=4, end='new record on the reasoning benchmarks among dense models.'),
    # ---------------------------------------------------------------- §1.1 Contributions, page 4 (index 3)
    dict(name='p6-contrib', page=3, anchor='Distillation: Smaller Models Can Be Powerful Too',
         highlight=['reasoning patterns of larger models can be distilled into smaller',
                    'better performance compared to the reasoning patterns discovered',
                    'DeepSeek-R1-Distill-Qwen-7B achieves 55.5% on AIME 2024'],
         above=4, end='32B, and 70B checkpoints based on Qwen2.5 and Llama3 series to the community.'),
    # ---------------------------------------------------------------- §2.3.3, the 800k samples, page 11 (index 10)
    dict(name='p6-800k', page=10, anchor='Non-Reasoning data',
         highlight=['approximately 200k training samples', 'about', '800k samples'],
         above=4, end='800k samples.'),
    # ---------------------------------------------------------------- §2.4, page 11 (index 10)
    dict(name='p6-s24-a', page=10, anchor='2.4. Distillation: Empower Small Models with Reasoning Capability',
         highlight=['directly', 'fine-tuned open-source models like Qwen', 'using',
                    'the 800k samples curated with DeepSeek-R1',
                    'this straightforward distillation method significantly enhances the reasoning abilities of smaller'],
         above=4, end='models. The base models we use here'),
    dict(name='p6-s24-b', page=10, anchor='The base models we use here are Qwen2.5-Math-1.5B',
         highlight=['Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5', '14B, Qwen2.5-32B, Llama-3.1-8B, and Llama-3.3-70B-Instruct',
                    'reasoning capability is slightly better than that of Llama-3.1'],
         above=4, end='reasoning capability is slightly better than that of Llama-3.1.'),
    dict(name='p6-s24-c', page=10, anchor='For distilled models, we apply only SFT',
         highlight=['we apply only SFT and do not include an RL stage', 'could substantially boost model performance',
                    'demonstrate the effectiveness of the distillation technique'],
         above=4, end='stage to the broader research community.'),
    # ---------------------------------------------------------------- §3, page 12 (index 11)
    dict(name='p6-bench', page=11, anchor='For distilled models, we',
         highlight=['For distilled models, we', 'report representative results on AIME 2024, MATH-500, GPQA Diamond, Codeforces, and',
                    'LiveCodeBench'],
         above=4, end='GPQA Diamond, Codeforces, and LiveCodeBench.'),
    dict(name='p6-baselines', page=11, anchor='Baselines',
         highlight=['GPT-4o-0513, OpenAI-o1-mini', 'For distilled models, we also compare the open-source model',
                    'QwQ-32B-Preview'],
         above=4, end='QwQ-32B-Preview (Qwen, 2024a).'),
    dict(name='p6-setup', page=11, anchor='Evaluation Setup',
         highlight=['maximum generation length to 32,768 tokens', 'sampling temperature of 0.6', 'top pvalue of 0.95',
                    'consensus (majority vote) results', 'cons@64'],
         above=4, end='denoted as cons@64.'),
    # ---------------------------------------------------------------- Table 5 and its text, page 14 (index 13)
    dict(name='p6-table5', page=13, anchor='3.2. Distilled Model Evaluation', column='full',
         highlight=['DeepSeek-R1-Distill-Qwen-1.5B', 'DeepSeek-R1-Distill-Qwen-32B', 'DeepSeek-R1-Distill-Llama-70B'],
         above=4, end='reasoning-related benchmarks.'),
    dict(name='p6-t5-text', page=13, anchor='As shown in Table 5, simply distilling',
         highlight=['outperform non', 'reasoning models like GPT-4o-0513 across the board',
                    'DeepSeek-R1-14B surpasses QwQ-32B', 'Preview on all evaluation metrics',
                    'exceed o1-mini on most benchmarks'],
         above=4, end='These results demonstrate the strong potential of distilla'),
    dict(name='p6-t5-rl', page=13, anchor='tion. Additionally, we found that applying RL',
         highlight=['applying RL to these distilled models yields significant further', 'gains',
                    'only the results of the', 'simple SFT-distilled models'],
         above=4, end='simple SFT-distilled models here.'),
    # ---------------------------------------------------------------- §4.1, page 14-15 (index 13, 14)
    dict(name='p6-s41-q', page=13, anchor='4.1. Distillation v.s. Reinforcement Learning',
         highlight=['can the model achieve comparable performance', 'through the large-scale RL training',
                    'without distillation', 'large-scale RL training on Qwen-32B-Base using math',
                    'code, and STEM data, training for over 10K steps', 'DeepSeek-R1-Zero-Qwen-32B'],
         above=4, end='training for over 10K steps, resulting in DeepSeek-R1-Zero-Qwen-32B'),
    dict(name='p6-table6', page=14, anchor='QwQ-32B-Preview', column='full',
         highlight=['DeepSeek-R1-Zero-Qwen-32B', 'DeepSeek-R1-Distill-Qwen-32B'],
         above=48, end='Comparison of distilled and RL Models on Reasoning-Related Benchmarks.'),
    dict(name='p6-t6-text', page=14, anchor='RL training, achieves performance on par with QwQ-32B-Preview',
         highlight=['achieves performance on par with QwQ-32B-Preview', 'performs significantly better than',
                    'across all benchmarks'],
         above=4, end='DeepSeek-R1-Zero-Qwen-32B across all benchmarks.'),
    dict(name='p6-conclusions', page=14, anchor='Therefore, we can draw two conclusions',
         highlight=['distilling more powerful models into smaller', 'ones yields excellent results',
                    'require enormous computational power and may not even achieve the performance',
                    'of distillation', 'advancing',
                    'beyond the boundaries of intelligence may still require more powerful base models and larger'],
         above=4, end='scale reinforcement learning.'),
    # ---------------------------------------------------------------- §5 Conclusion, page 16 (index 15)
    dict(name='p6-conclusion', page=15, anchor='We further explore distillation the reasoning capability',
         highlight=['800K training samples', 'on math benchmarks with 28.9% on AIME and 83.9% on MATH'],
         above=4, end='tuned models based on the same underlying checkpoints.'),
    # ---------------------------------------------------------------- the later long version, Appendix B.4.3
    dict(name='p6-hparams', arxiv_id=LONG, page=34, anchor='B.4.3. Hyper-Parameters of Distillation',
         highlight=['2–3 epochs', 'cosine decay learning rate scheduler', 'one-tenth of its initial value',
                    'batch size is 64', 'Qwen2.5-Math-1.5B', '1 × 10−4'],
         hl_nth={'2–3 epochs': 1, 'cosine decay learning rate scheduler': 1}, above=4, end='2 × 10−5'),
    # ---------------------------------------------------------------- earlier papers
    dict(name='p6-hinton', arxiv_id=HINTON, page=1, anchor='An obvious way to transfer the generalization ability',
         highlight=['use the class probabilities produced by the cumbersome model as', 'soft targets'],
         above=4, end='less data than the original cumbersome model and using a much higher learning rate.'),
    dict(name='p6-hinton-bmw', arxiv_id=HINTON, page=1, anchor='For cumbersome models that learn to discriminate',
         highlight=['relative probabilities of incorrect answers', 'An image of a BMW', 'mistaken for a garbage truck'],
         above=4, end='probable than mistaking it for a carrot.'),
    dict(name='p6-kimrush', arxiv_id=KIMRUSH, page=1, anchor='knowledge distillation for NMT and also propose',
         highlight=['sequence-level (as opposed', 'to word-level)', 'trained on a newly generated dataset that is the', 'result of running beam search with the teacher network'],
         above=4, end='result of running beam search with the teacher network'),
]

if __name__ == '__main__':
    run(JOBS)
