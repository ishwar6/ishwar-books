"""Highlighted excerpts of the Transformer paper for Part 1 (the big idea): title, abstract, §1 and §2."""
from paper_shots import run

JOBS = [
    # ---------------------------------------------------------------- page 1: title and abstract
    dict(name='p1-title', page=0, anchor='Attention Is All You Need', highlight=['Attention Is All You Need'],
         column='full', above=8, below=8),
    dict(name='p1-abstract-1', page=0, anchor='The dominant sequence transduction models',
         highlight=['sequence transduction models', 'recurrent or convolutional neural networks', 'an encoder and a decoder',
                    'through an attention mechanism'],
         above=4, end='attention mechanism. We propose'),
    dict(name='p1-abstract-2', page=0, anchor='We propose a new simple network architecture',
         highlight=['the Transformer', 'based solely on attention mechanisms', 'dispensing with recurrence and convolutions',
                    'more parallelizable', 'significantly less time to train'],
         above=4, end='less time to train.'),
    dict(name='p1-abstract-3', page=0, anchor='Our model achieves 28.4 BLEU',
         highlight=['28.4 BLEU', 'by over 2 BLEU', '41.8', '3.5 days on eight GPUs', 'small fraction of the training costs'],
         above=4, end='from the literature.'),
    dict(name='p1-abstract-4', page=0, anchor='We show that the Transformer generalizes well',
         highlight=['generalizes well', 'English constituency parsing'], above=4, end='limited training data.'),
    # ---------------------------------------------------------------- page 2: §1 Introduction
    dict(name='p1-intro-1', page=1, anchor='Recurrent neural networks, long short-term memory',
         highlight=['Recurrent neural networks', 'long short-term memory [13]', 'gated recurrent [7]',
                    'language modeling and machine translation'],
         above=18, end='architectures [38, 24, 15].'),
    dict(name='p1-intro-2', page=1, anchor='Recurrent models typically factor computation',
         highlight=['hidden', 'states ht, as a function of the previous hidden state ht−1 and the input for position t',
                    'inherently', 'sequential nature precludes parallelization within training examples',
                    'memory constraints limit batching across examples', 'fundamental', 'constraint of sequential computation'],
         above=4, end='computation, however, remains.'),
    dict(name='p1-intro-3', page=1, anchor='Attention mechanisms have become an integral part',
         highlight=['without regard to their distance', 'used in conjunction with a recurrent network'],
         above=4, end='with a recurrent network.'),
    dict(name='p1-intro-4', page=1, anchor='In this work we propose the Transformer',
         highlight=['eschewing recurrence', 'relying entirely on an attention mechanism', 'global dependencies',
                    'twelve hours on eight P100 GPUs'],
         above=4, end='eight P100 GPUs.'),
    # ---------------------------------------------------------------- page 2: §2 Background
    dict(name='p1-background-1', page=1, anchor='The goal of reducing sequential computation',
         highlight=['Extended Neural GPU', 'ByteNet [18]', 'ConvS2S [9]', 'computing hidden representations in parallel',
                    'linearly for ConvS2S and logarithmically for ByteNet', 'constant number of operations',
                    'reduced effective resolution', 'Multi-Head Attention'],
         above=18, end='in section 3.2.'),
    dict(name='p1-background-2', page=1, anchor='Self-attention, sometimes called intra-attention',
         highlight=['relating different positions', 'of a single sequence', 'compute a representation of the sequence'],
         above=4, end='representations [4, 27, 28, 22].'),
    dict(name='p1-background-3', page=1, anchor='To the best of our knowledge',
         highlight=['first transduction model relying', 'entirely on self-attention', 'without using sequence-',
                    'aligned RNNs or convolution'],
         above=4, end='and [9].'),
]

if __name__ == '__main__':
    run(JOBS)
    # page 1 has one narrow column: cut away the empty margin and arXiv's side stamp on the left
    import os
    from PIL import Image
    from paper_shots import OUT
    for j in JOBS:
        if j['page'] == 0 and j['name'] != 'p1-title':
            p = os.path.join(OUT, j['name'] + '.png')
            im = Image.open(p)
            im.crop((300, 0, im.width - 90, im.height)).save(p)
