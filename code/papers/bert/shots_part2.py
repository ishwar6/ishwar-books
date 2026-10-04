"""Highlighted excerpts of the BERT paper for Part 2 (Section 2, Section 3 model and input, Figures 1 and 2)."""
from paper_shots import run

JOBS = [
    dict(name='p2-related-intro', page=1, anchor='There is a long history of pre-training general lan',
         highlight=['There is a long history of pre-training general language representations'], end='most widely-used approaches in this section.', above=24),
    dict(name='p2-feature-words', page=1, anchor='Learning widely applicable representations of',
         highlight=['Pre-trained word embeddings', 'significant improvements over embeddings learned from scratch'],
         end='right context (Mikolov et al., 2013).', above=18),
    dict(name='p2-feature-sentences', page=1, anchor='These approaches have been generalized to',
         highlight=['sentence embeddings', 'paragraph embeddings'], end='encoder derived objectives (Hill et al., 2016).', above=4),
    dict(name='p2-elmo', page=1, anchor='ELMo and its predecessor',
         highlight=['context-sensitive features', 'is the concatenation of the left-to-right and right-to-left representations',
                    'not deeply bidirectional'], end='to improve the robustness of text generation models.', above=4),
    dict(name='p2-finetune', page=1, anchor='As with the feature-based approaches, the first',
         highlight=['only pre-trained word embedding parameters', 'fine-tuned for a supervised downstream task',
                    'few parameters need to be learned from scratch'], end='GLUE benchmark (Wang et al., 2018a).', above=30),
    dict(name='p2-finetune2', page=2, anchor='ing and auto-encoder objectives have been used',
         highlight=['auto-encoder objectives'], end='Dai and Le, 2015).', above=4),
    dict(name='p2-supervised', page=2, anchor='There has also been work showing effective trans',
         highlight=['natural language inference', 'machine translation', 'fine-tune models pre-trained with ImageNet'],
         end='Yosinski et al., 2014).', above=20),
    dict(name='p2-figure1', page=2, anchor='Figure 1: Overall pre-training and fine-tuning procedures for BERT',
         highlight=['Apart from output layers, the same architectures are used in both pre-training and fine-tuning',
                    'During fine-tuning, all parameters are fine-tuned'], figure=True, column='full', below=0),
    dict(name='p2-bert-intro', page=2, anchor='We introduce BERT and its detailed implementa',
         highlight=['There are two steps in our framework', 'all of the parameters are fine-tuned using labeled data from the downstream tasks',
                    'Each downstream task has separate fine-tuned models'], end='running example for this section.', above=24),
    dict(name='p2-unified', page=2, anchor='A distinctive feature of BERT is its unified ar',
         highlight=['A distinctive feature of BERT is its unified architecture across different tasks'], above=4, below=16),
    dict(name='p2-architecture', page=2, anchor='mal difference between the pre-trained architec',
         highlight=['difference between the pre-trained architecture and the final downstream architecture', 'multi-layer bidirectional Transformer encoder', 'almost identical to the original'],
         end='as well as excellent guides such as The Annotated', above=4),
    dict(name='p2-sizes', page=2, anchor='In this work, we denote the number of layers',
         highlight=['L=12, H=768, A=12, Total Parameters=110M', 'L=24, H=1024, A=16, Total Parameters=340M',
                    'same model size as OpenAI GPT', 'bidirectional self-attention', 'can only attend to context to its'],
         end='token can only attend to context to its', above=4),
    dict(name='p2-footnote3', page=2, anchor='we set the feed-forward/filter size to be 4H',
         highlight=['feed-forward/filter size to be 4H', '3072 for the H = 768 and 4096 for the H = 1024'], above=4, below=10),
    dict(name='p2-footnote4a', page=2, anchor='note that in the literature the bidirectional Trans',
         highlight=['bidirectional Trans'], above=4, below=1),
    dict(name='p2-footnote4', page=3, anchor='former is often referred to as a “Transformer encoder” while',
         highlight=['Transformer encoder', 'Transformer decoder'], end='it can be used for text generation.', above=4),
    dict(name='p2-input', page=3, anchor='Input/Output Representations',
         highlight=['a “sentence” can be an arbitrary span of contiguous text', 'A “sequence” refers to the input token sequence to BERT'],
         end='or two sentences packed together.', above=4),
    dict(name='p2-wordpiece', page=3, anchor='We use WordPiece embeddings',
         highlight=['WordPiece embeddings', '30,000 token vocabulary', '([CLS])', 'aggregate sequence representation', '([SEP])',
                    'a learned embedding to every token indicating whether it belongs to sentence A or sentence B'],
         end='as Ti ∈ RH.', above=4),
    dict(name='p2-sum', page=3, anchor='For a given token, its input representation is',
         highlight=['summing the corresponding token, segment, and position embeddings'], end='can be seen in Figure 2.', above=4),
    dict(name='p2-figure2', page=4, anchor='Figure 2: BERT input representation',
         highlight=['The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings'],
         figure=True, column='full', below=0),
    # ---- v2: the tensor2tensor and Annotated Transformer footnotes, and excerpts from the papers Section 2 and 3 cite
    dict(name='p2-footnotes12', page=2, anchor='1https://github.com/tensorflow/tensor2tensor',
         highlight=['github.com/tensorflow/tensor2tensor', 'nlp.seas.harvard.edu/2018/04/03/attention.html'],
         end='2http://nlp.seas.harvard.edu/2018/04/03/attention.html', above=4),
    dict(name='p2-ext-skipgram-fig1', arxiv_id='1310.4546', page=1, anchor='Figure 1: The Skip-gram model architecture.',
         highlight=['good at predicting the nearby words'], figure=True, column='full', below=0),
    dict(name='p2-ext-skipgram-neg', arxiv_id='1310.4546', page=2, anchor='We define Negative sampling (NEG) by the objective',
         highlight=['Negative sampling (NEG)'], above=30, below=40),
    dict(name='p2-ext-skipthought-fig1', arxiv_id='1506.06726', page=1, anchor='Figure 1: The skip-thoughts model.',
         highlight=['the sentence si is encoded and tries to reconstruct the previous sentence si−1 and next sentence si+1'], figure=True, column='full', below=0),
    dict(name='p2-ext-quickthought-fig1', arxiv_id='1803.02893', page=2, anchor='Figure 1: Overview.',
         highlight=['replaces the decoder with a classifier which chooses the target sentence'], figure=True, column='full', below=0),
    dict(name='p2-ext-hill-sdae', arxiv_id='1602.03483', page=2, anchor='representation-learning objective based on denoising autoencoders',
         highlight=['N deletes w with (independent) probability po', 'swaps wi and wi+1 with probability px', 'predict (as target) the original source sentence'],
         end='quences into distributed representations.', above=4),
    dict(name='p2-ext-vaswani-fig1', arxiv_id='1706.03762', page=2, anchor='Figure 1: The Transformer - model architecture.',
         highlight=['The Transformer - model architecture'], figure=True, column='full', below=0),
    dict(name='p2-ext-vaswani-eq1', arxiv_id='1706.03762', page=3, anchor='In practice, we compute the attention function on a set',
         highlight=['packed together into a matrix Q', 'We compute the matrix of outputs as:'], end='Dot-product attention is identical to our algorithm', above=4),
    dict(name='p2-ext-vaswani-mha', arxiv_id='1706.03762', page=4, anchor='Multi-head attention allows the model to jointly attend',
         highlight=['jointly attend to information from different representation subspaces at different positions'],
         end='In this work we employ h = 8 parallel attention layers', above=4),
    dict(name='p2-ext-vaswani-ffn', arxiv_id='1706.03762', page=4, anchor='In addition to attention sub-layers, each of the layers',
         highlight=['applied to each position separately and identically', 'two linear transformations with a ReLU activation in between'],
         end='Another way of describing this is as two convolutions', above=4),
    dict(name='p2-ext-wu-wordpiece', arxiv_id='1609.08144', page=6, anchor='Here is an example of a word sequence and the corresponding',
         highlight=['wordpieces:', 'is a special character added to mark the beginning of a word'], end='character added to mark the beginning of a word.', above=4),
]



def trim_peek(name, limit=24):
    """Cut off a sliver of the previous line that peeks in at the top of a tightly spaced crop."""
    import os
    from PIL import Image
    from paper_shots import OUT
    p = os.path.join(OUT, name + '.png')
    im = Image.open(p).convert('L')
    w, h = im.size
    rows = [min(im.getpixel((x, y)) for x in range(0, w, 2)) for y in range(limit)]
    if rows[0] < 200:                                   # ink touches the top edge: a clipped line
        gap = next((y for y, v in enumerate(rows) if v > 245), None)
        if gap:
            Image.open(p).crop((0, gap, w, h)).save(p)
            print(f'{name}: trimmed {gap}px peek at the top')


if __name__ == '__main__':
    run(JOBS)
    import os
    if not os.environ.get('ONLY') or 'p2-ext-hill-sdae' in os.environ['ONLY']:
        trim_peek('p2-ext-hill-sdae')
