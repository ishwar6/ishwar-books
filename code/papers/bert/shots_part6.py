"""Highlighted excerpts of the BERT paper for Part 6 (conclusion, and the lines behind each limitation)."""
from paper_shots import run

JOBS = [
    dict(name='p6-conclusion', page=8, anchor='Recent empirical improvements due to transfer',
         highlight=['rich, unsupervised pre-training is an integral part of many language understanding systems',
                    'even low-resource tasks to benefit from deep unidirectional architectures',
                    'Our major contribution is further generalizing these findings to deep bidirectional architectures'],
         end='successfully tackle a broad set of NLP tasks.', above=4),
    dict(name='p6-mismatch', page=3, anchor='Although this allows us to obtain a bidirec',
         highlight=['a downside is that we are creating a mismatch between pre-training and fine-tuning',
                    'does not appear during fine-tuning'], end='does not appear during fine-tuning.', above=4),
    dict(name='p6-fifteen', page=11, anchor='Compared to standard langauge model training',
         highlight=['only make predictions on 15% of tokens in each batch', 'more pre-training steps may be required'],
         end='more pre-training steps may be required for the model', above=4),
    dict(name='p6-footnote6', page=3, anchor='vector C is not a meaningful sentence representation',
         highlight=['vector C is not a meaningful sentence representation without fine-tuning'], end='since it was trained with NSP.', above=4),
    dict(name='p6-compute', page=12, anchor='was performed on 4 Cloud TPUs in Pod',
         highlight=['4 Cloud TPUs in Pod configuration (16 TPU chips', '16 Cloud TPUs (64 TPU chips', 'Each pre-training took 4 days to complete'],
         end='Each pre-training took 4 days to complete.', above=4),
    dict(name='p6-length', page=12, anchor='Longer sequences are disproportionately expen',
         highlight=['attention is quadratic to the sequence length'], end='to learn the positional embeddings.', above=4),
]

if __name__ == '__main__':
    run(JOBS)
