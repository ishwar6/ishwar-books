"""Highlighted excerpts of the paper for Part 5 (training): Section 5 on pages 7 and 8."""
from paper_shots import run

JOBS = [
    dict(name='p5-data', page=6, anchor='Training Data and Batching',
         highlight=['about 4.5 million sentence pairs', 'byte-pair encoding', 'shared source-target vocabulary of about 37000 tokens',
                    '36M sentences', '32000 word-piece vocabulary'],
         above=4, end='vocabulary [38].'),
    dict(name='p5-batching', page=6, anchor='Sentence pairs were batched together',
         highlight=['batched together by approximate sequence length', 'approximately 25000 source tokens and 25000 target tokens'],
         above=4, end='target tokens.'),
    dict(name='p5-hardware', page=6, anchor='Hardware and Schedule',
         highlight=['8 NVIDIA P100 GPUs', 'about 0.4 seconds', '100,000 steps or 12 hours', '300,000 steps', '(3.5 days)'],
         above=4, end='(3.5 days).'),
    dict(name='p5-adam', page=6, anchor='We used the Adam optimizer',
         highlight=['Adam optimizer', 'β1 = 0.9, β2 = 0.98'],
         above=20, end='according to the formula:'),
    dict(name='p5-eq3', page=6, anchor='according to the formula:',
         highlight=['increasing the learning rate linearly for the first warmup_steps',
                    'inverse square root of the step number', 'warmup_steps = 4000'],
         above=4, end='warmup_steps = 4000.', band=(13, 37)),
    dict(name='p5-dropout', page=7, anchor='Residual Dropout We apply dropout',
         highlight=['output of each sub-layer, before it is added to the', 'sums of the embeddings and the',
                    'Pdrop = 0.1'],
         above=24, end='Pdrop = 0.1.'),
    dict(name='p5-smoothing', page=7, anchor='Label Smoothing During training',
         highlight=['label smoothing of value', 'hurts perplexity', 'more unsure', 'improves accuracy and BLEU score'],
         above=4, end='BLEU score.'),
]

if __name__ == '__main__':
    run(JOBS)
