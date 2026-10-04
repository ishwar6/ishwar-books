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
    dict(name='p6-footnote4', page=3, anchor='former is often referred to as a',
         highlight=['Transformer decoder', 'since it can be used for text generation'], end='since it can be used for text generation.', above=4, below=10),
    dict(name='p6-finetune-cost', page=4, anchor='Compared to pre-training, fine-tuning is rela',
         highlight=['can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU', 'starting from the exact same pre-trained'],
         end='from the exact same pre-trained', above=4),
    dict(name='p6-roberta', arxiv_id='1907.11692', page=0, anchor='Language model pretraining has led to',
         highlight=['We find that BERT was significantly undertrained', 'can match or exceed the performance of every model published after it'],
         end='We release our models and code.1', above=4),
    dict(name='p6-xlnet', arxiv_id='1906.08237', page=0, anchor='With the capability of modeling bidirectional',
         highlight=['BERT neglects dependency between the masked positions', 'suffers from a pretrain-finetune discrepancy'],
         end='inference, sentiment analysis, and document', above=4),
    dict(name='p6-albert', arxiv_id='1909.11942', page=0, anchor='Increasing model size when pretraining natural',
         highlight=['two parameter-reduction techniques', 'self-supervised loss that focuses on modeling inter-sentence coherence'],
         end='https://github.com/google-research/ALBERT.', above=4),
    dict(name='p6-distilbert', arxiv_id='1910.01108', page=0, anchor='As Transfer Learning from large-scale pre-trained',
         highlight=['knowledge distillation during the pre-training phase', 'reduce the size of a BERT model by 40%, while retaining 97%',
                    'being 60% faster'], end='comparative on-device study.', above=4),
    dict(name='p6-electra', arxiv_id='2003.10555', page=0, anchor='Masked language modeling (MLM) pre-training methods',
         highlight=['replaced token detection', 'the task is defined over all input tokens rather than just the small subset that was masked out'],
         end='when using the same amount of compute.', above=4),
    dict(name='p6-sbert', arxiv_id='1908.10084', page=0, anchor='BERT (Devlin et al., 2018) and RoBERTa',
         highlight=['requires about 50 million inference computations', 'semantically meaningful sentence embeddings', 'compared using cosine-similarity'],
         end='sentence embeddings methods.1', above=4),
    dict(name='p6-modernbert', arxiv_id='2412.13663', page=0, anchor='Encoder-only transformer models such as',
         highlight=['the workhorse of numerous production pipelines', '2 trillion tokens with a native 8192 sequence length'],
         end='designed for inference on common GPUs.', above=4),
]

STAMPED = ['p6-xlnet', 'p6-albert', 'p6-distilbert', 'p6-electra']   # one-column papers: the crop catches arXiv's rotated side stamp


def trim_stamp(name):
    """Cut away the rotated arXiv stamp at the left edge: drop everything left of the wide white gap before the text."""
    import os
    from PIL import Image
    from paper_shots import OUT
    p = os.path.join(OUT, name + '.png')
    im = Image.open(p).convert('RGB')
    g = im.convert('L').point(lambda v: 255 if v < 200 else 0)
    W, H = im.size
    ink = [g.crop((x, 0, x + 1, H)).getbbox() is not None for x in range(W)]
    x, gap = 0, 0
    while x < W and not ink[x]:
        x += 1
    while x < W and ink[x]:                 # the stamp
        x += 1
    start = x
    while x < W and not ink[x]:             # the white gap
        x += 1
    if x - start > 60:
        right = max(i for i in range(W) if ink[i])
        im.crop((max(0, x - 24), 0, min(W, right + 24), H)).save(p)
        print(f'{name}: trimmed the side stamp, now {min(W, right + 24) - max(0, x - 24)}x{H}')


if __name__ == '__main__':
    import os
    run(JOBS)
    for n in STAMPED:
        if not os.environ.get('ONLY') or n in os.environ['ONLY'].split(','):
            trim_stamp(n)
