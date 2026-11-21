"""Highlighted excerpts for Part 7: §3 (benchmarks, prompts, baselines, setup), §3.1 and Table 4, §4.2 (PRM, MCTS),
§5 (conclusion and limits); plus Chen et al. (2021) for pass@k, AlphaGo (2016) for the size of Go, and the Nature
version of the paper for what came after. Pages are 0-based."""
from paper_shots import run

JOBS = [
    # §3 Experiment
    dict(name='p7-benchmarks', page=10, anchor='Benchmarks We evaluate models on MMLU',
         highlight=['MMLU (Hendrycks et al., 2020)', 'GPQA Diamond (Rein et al.,', 'SimpleQA (OpenAI, 2024c)',
                    'SWE-Bench Verified (OpenAI,'],
         above=4, end='SWE-Bench Verified (OpenAI,', below=8, column='full'),
    dict(name='p7-judges', page=11, anchor='2024d), Aider',
         highlight=['LiveCodeBench (Jain et al., 2024)', 'Codeforces', 'AIME 2024', 'using LLMs as judges',
                    'we only feed the final summary to evaluation to avoid the length bias'],
         above=4, end='and LiveCodeBench.', column='full'),
    dict(name='p7-prompts', page=11, anchor='Evaluation Prompts Following the setup',
         highlight=['in a zero-shot setting', 'we slightly modify the prompt to the zero-shot setting',
                    'The CoT in few-shot may hurt the performance of DeepSeek-R1'],
         above=4, end='may hurt the performance of DeepSeek-R1.', column='full'),
    dict(name='p7-code-setup', page=11, anchor='Other datasets follow their original evaluation',
         highlight=['10 Div.2 contests', 'expected ratings and percentages of competitors', 'agentless framework',
                    'capped at a maximum of 32,768 tokens'],
         above=4, end='32,768 tokens for each benchmark.', column='full'),
    dict(name='p7-baselines', page=11, anchor='Baselines We conduct comprehensive evaluations',
         highlight=['challenging in mainland China', 'based on official reports'],
         above=4, end='QwQ-32B-Preview (Qwen, 2024a).', column='full'),
    dict(name='p7-setup', page=11, anchor='Evaluation Setup We set the maximum generation length',
         highlight=['greedy decoding to evaluate long-output reasoning models results in',
                    'higher repetition rates and significant variability across different checkpoints',
                    'sampling temperature of 0.6',
                    'typically between 4 and 64'],
         above=4, end='for each question. Pass@1', column='full'),
    dict(name='p7-pass1', page=11, anchor='is then calculated as',
         highlight=['correctness of the i-th response', 'cons@64'],
         above=4, end='denoted as cons@64.', column='full'),
    dict(name='p7-chen', arxiv_id='2107.03374', page=2, anchor='Instead, to evaluate pass@k',
         highlight=['count the number of correct', 'unbiased estimator', 'it is biased'],
         above=40, end='biased in Appendix A.'),
    dict(name='p7-chen-code', arxiv_id='2107.03374', page=2, anchor='def pass_at_k(n, c, k):',
         highlight=['if n - c < k: return 1.0', 'unbiased estimate of pass@k'], above=6, end='unbiased estimate of pass@k.', column='full'),
    # §3.1 and Table 4
    dict(name='p7-table4', page=12, anchor='Benchmark (Metric)',
         highlight=['IF-Eval (Prompt Strict)', 'SimpleQA (Correct)', 'Aider-Polyglot (Acc.)', 'C-SimpleQA (Correct)'],
         above=26, end='Table 4 | Comparison between DeepSeek-R1', column='full'),
    dict(name='p7-knowledge', page=12, anchor='For education-oriented knowledge benchmarks',
         highlight=['enhanced accuracy in STEM-related questions', 'excels on FRAMES',
                    'its tendency to refuse', 'answering certain queries after safety RL',
                    'accuracy of over 70%'],
         above=4, end='accuracy of over 70%.', column='full'),
    dict(name='p7-ifeval', page=12, anchor='DeepSeek-R1 also delivers impressive results on IF-Eval',
         highlight=['instruction-following data during the final stages', 'generalization benefits of large-scale',
                    'average of 689 tokens on ArenaHard and 2,218 characters on AlpacaEval 2.0'],
         above=4, end='This indicates that', column='full'),
    dict(name='p7-math-code', page=13, anchor='On math tasks, DeepSeek-R1 demonstrates',
         highlight=['outperforms DeepSeek-R1', 'on Aider but achieves comparable performance on SWE Verified',
                    'related RL training data currently remains very limited'],
         above=4, end='remains very limited.', column='full'),
    # §4.2 Unsuccessful attempts
    dict(name='p7-unsuccessful', page=14, anchor='4.2. Unsuccessful Attempts',
         highlight=['this does not imply that these approaches are incapable'],
         above=4, end='effective reasoning models.', column='full'),
    dict(name='p7-prm', page=14, anchor='Process Reward Model (PRM) PRM is a reasonable',
         highlight=['define a fine-grain step in general reasoning',
                    'determining whether the current intermediate step is correct',
                    'inevitably leads to reward', 'rerank the top-N responses'],
         above=4, end='in our experiments.', column='full'),
    dict(name='p7-mcts-a', page=14, anchor='Monte Carlo Tree Search (MCTS) Inspired by AlphaGo',
         highlight=['breaking answers into smaller parts', 'multiple tags', 'guided by a pre-trained value',
                    'unlike chess, where the search space is relatively well-defined'],
         above=4, end='token generation presents an', column='full'),
    dict(name='p7-mcts-b', page=15, anchor='exponentially larger search space. To address this',
         highlight=['maximum extension limit for each', 'stuck in local optima',
                    'Training a fine-grained value model is inherently difficult',
                    'iteratively boosting model performance through self-search remains a'],
         above=4, end='significant challenge.', column='full'),
    dict(name='p7-alphago', arxiv_id='alphago2016', page=0, anchor='by recursively computing the optimal value function',
         highlight=['chess (b≈35, d≈80)1', 'Go (b≈250, d≈150)1,', 'exhaustive search is'],
         above=3, end='two general principles.'),
    # §5
    dict(name='p7-conclusion', page=15, anchor='5. Conclusion, Limitations, and Future Work',
         highlight=['comparable to OpenAI-o1-1217 on a range of tasks', '800K training samples',
                    '28.9% on AIME and 83.9% on MATH'],
         above=4, end='same underlying checkpoints.', column='full'),
    dict(name='p7-limits-a', page=15, anchor='In the future, we plan to invest in research',
         highlight=['function calling, multi-turn, complex role-playing, and JSON output',
                    'optimized for Chinese and English', 'might use English for reasoning and responses'],
         above=4, end='in future updates.', column='full'),
    dict(name='p7-limits-b', page=15, anchor='Prompting Engineering: When evaluating',
         highlight=['Few-shot prompting consistently degrades its performance', 'zero-shot setting',
                    'long evaluation times', 'rejection sampling on software engineering data',
                    'asynchronous evaluations'],
         above=4, end='to improve efficiency.', column='full'),
    # the later Nature version
    dict(name='p7-nature-limits', arxiv_id='nature-r1', page=3, anchor='Structure output and tool use',
         highlight=['cannot make use of tools', 'Token efficiency', 'overthinking'],
         above=4, end='to simpler questions.'),
]



def right_column(name, x0=680):
    """The Chen et al. code sits in the right column; a full-width crop also catches the left one, so cut it away."""
    import os
    from PIL import Image
    from paper_shots import OUT
    path = os.path.join(OUT, name + '.png')
    im = Image.open(path)
    im.crop((x0, 0, im.width, im.height)).save(path)
    print(f'{name}: kept the right column, x >= {x0}')


if __name__ == '__main__':
    run(JOBS)
    import os
    if not os.environ.get('ONLY') or 'p7-chen-code' in os.environ['ONLY']:
        right_column('p7-chen-code')
