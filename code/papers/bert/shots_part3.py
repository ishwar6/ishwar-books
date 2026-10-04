"""Highlighted excerpts of the BERT paper for Part 3 (Section 3.1, Appendix A.1 and A.2)."""
from paper_shots import run

JOBS = [
    dict(name='p3-intro', page=3, anchor='Unlike Peters et al. (2018a) and Radford et al.',
         highlight=['we do not use traditional left-to-right or right-to-left language models', 'two unsupervised tasks'],
         end='This step is presented in the left part of Figure 1.', above=22),
    dict(name='p3-mlm-why', page=3, anchor='Intuitively, it is reason',
         highlight=['strictly more powerful', 'bidirectional conditioning would allow each word to indirectly',
                    'see itself', 'trivially predict the target word in a multi-layered context'],
         end='predict the target word in a multi-layered context.', above=16),
    dict(name='p3-mlm-how', page=3, anchor='In order to train a deep bidirectional representa',
         highlight=['we simply mask some percentage of the input tokens at random', 'fed into an output softmax over the vocabulary',
                    'we mask 15% of all WordPiece tokens in each sequence at random', 'only predict the masked words rather than reconstructing the entire input'],
         end='reconstructing the entire input.', above=4),
    dict(name='p3-mlm-8010', page=3, anchor='Although this allows us to obtain a bidirec',
         highlight=['a mismatch between pre-training and fine-tuning', 'chooses 15% of the token positions at random',
                    'the [MASK] token 80% of the time', 'a random token 10% of the time', 'the unchanged i-th token 10% of the time',
                    'will be used to predict the original token with cross entropy loss'],
         end='procedure in Appendix C.2.', above=4),
    dict(name='p3-nsp', page=3, anchor='Many important downstream tasks such as Ques',
         highlight=['understanding the relationship between two sentences', 'not directly captured by language modeling',
                    '50% of the time B is the actual next sentence that follows A', '50% of the time it is a random sentence from the corpus',
                    'is used for next sentence prediction'],
         end='beneficial to both QA and NLI.', above=22),
    dict(name='p3-footnotes', page=3, anchor='final model achieves 97%-98% accuracy on NSP',
         highlight=['final model achieves 97%-98% accuracy on NSP', 'vector C is not a meaningful sentence representation without fine-tuning'],
         end='since it was trained with NSP.', above=4),
    dict(name='p3-nsp-prior', page=4, anchor='The NSP task is closely related to representation',
         highlight=['only sentence embeddings are transferred to down-stream tasks', 'BERT transfers all parameters'],
         end='initialize end-task model parameters.', above=4),
    dict(name='p3-data', page=4, anchor='Pre-training data The pre-training procedure',
         highlight=['BooksCorpus (800M words)', 'English Wikipedia (2,500M words)', 'we extract only the text passages and ignore lists, tables, and headers',
                    'critical to use a document-level corpus rather than a shuffled sentence-level corpus'],
         end='to extract long contiguous sequences.', above=4),
    dict(name='p3-a1-mlm', page=11, anchor='Masked LM and the Masking Procedure',
         highlight=['80% of the time: Replace the word with the [MASK] token', '10% of the time: Replace the word with a random word',
                    'Keep the word unchanged', 'bias the representation towards the actual observed word'],
         end='The purpose of this is to bias the representation towards the actual observed word.', above=20),
    dict(name='p3-a1-why', page=11, anchor='The advantage of this procedure is that the',
         highlight=['does not know which words it will be asked to predict', 'distributional contextual representation of every input token',
                    'random replacement only occurs for 1.5% of all tokens (i.e., 10% of 15%)'],
         end='we evaluate the impact this procedure.', above=4),
    dict(name='p3-a1-cost', page=11, anchor='Compared to standard langauge model training',
         highlight=['only make predictions on 15% of tokens in each batch'], end='more pre-training steps may be required for the model', above=4),
    dict(name='p3-a1-cost2', page=12, anchor='to converge. In Section C.1 we demonstrate that',
         highlight=['MLM does converge marginally slower than a left-to-right model', 'far outweigh the increased training cost'],
         end='far outweigh the increased training cost.', above=4),
    dict(name='p3-a1-nsp', page=12, anchor='Next Sentence Prediction The next sentence',
         highlight=['Label = IsNext', 'Label = NotNext'], end='Label = NotNext', above=4),
    dict(name='p3-a2-data', page=12, anchor='To generate each training input sequence, we sam',
         highlight=['sample two spans of text from the corpus', 'combined length is', '512 tokens',
                    'after WordPiece tokenization with a uniform masking rate of 15%', 'no special consideration given to partial word pieces'],
         end='no special consideration given to partial word pieces.', above=20),
    dict(name='p3-a2-train0', page=12, anchor='We train with batch size of 256 sequences',
         highlight=['batch size of 256 sequences', '256 sequences * 512 tokens = 128,000 tokens/batch', '1,000,000 steps', 'approximately 40'],
         end='for 1,000,000 steps, which is approximately 40', above=4),
    dict(name='p3-a2-train', page=12, anchor='epochs over the 3.3 billion word corpus',
         highlight=['epochs over the 3.3 billion word corpus',
                    'Adam with learning rate of 1e-4', 'L2 weight decay of 0.01', 'warmup over the first 10,000 steps', 'linear decay of the learning rate',
                    'dropout probability of 0.1 on all layers', 'gelu activation',
                    'sum of the mean masked LM likelihood and the mean next sentence prediction likelihood'],
         end='likelihood and the mean next sentence prediction likelihood.', above=4),
    dict(name='p3-a2-tpu', page=12, anchor='Training of BERTBASE was performed on 4',
         highlight=['16 TPU chips', '64 TPU chips total', 'Each pre-training took 4 days to complete',
                    'attention is quadratic to the sequence length', 'sequence length of 128 for 90% of the steps', 'the rest 10% of the steps of sequence of 512'],
         end='to learn the positional embeddings.', above=4),
    dict(name='p3-figure1', page=2, anchor='Figure 1: Overall pre-training and fine-tuning procedures for BERT',
         highlight=['the same architectures are used in both pre-training and fine-tuning'], figure=True, column='full', below=0),
    dict(name='p3-gelu-paper', arxiv_id='1606.08415', page=0, anchor='We propose the Gaussian Error Linear Unit',
         highlight=['The GELU activation function is', 'the standard Gaussian cumulative distribution function', 'weights inputs by their value, rather than gates inputs by their sign'],
         end='rather than gates inputs by their sign as in ReLUs', above=4),
    dict(name='p3-billion', arxiv_id='1312.3005', page=1, anchor='Because the original data had already randomized',
         highlight=['already randomized sentence order', 'not useful for experiments with models that capture long context dependencies across sentence boundaries'],
         end='dependencies across sentence boundaries.', above=4),
    dict(name='p3-billion-steps', arxiv_id='1312.3005', page=1, anchor='Sentence order was randomized',
         highlight=['Sentence order was randomized'], end='into 100 disjoint partitions', above=4),
    dict(name='p3-bookcorpus', arxiv_id='1506.06724', page=3, anchor='The learning signal of the model depends on having',
         highlight=['depends on having contiguous text, where sentences follow one another in sequence', 'a large collection of books'],
         end='a large collection of books.', above=4),
    dict(name='p3-bookcorpus-table', arxiv_id='1506.06724', page=2, anchor='Table 2: Summary statistics of our BookCorpus dataset',
         highlight=['11,038', '984,846,357'], figure=True, fig_top=30, below=0),
    dict(name='p3-logeswaran', arxiv_id='1803.02893', page=0, anchor='In this work we propose a simple and efficient framework',
         highlight=['reformulate the problem of predicting the context in which a sentence appears as a classification problem',
                    'distinguishes context sentences from other contrastive sentences'],
         end='sentences from other contrastive sentences based on their vector representations.', above=4),
    dict(name='p3-jernite', arxiv_id='1705.00557', page=2, anchor='Many coherence relations are',
         highlight=['two adjacent sentences will generally be more coherent than two more distant ones', 'decide which candidate immediately follows the initial three'],
         end='immediately follows the initial three in the source text.', above=14),
]

# crops of other papers' abstracts start in the left margin, next to arXiv's rotated stamp: trim it (pixels at 190 dpi)
TRIM = {'p3-gelu-paper': (300, 0), 'p3-logeswaran': (300, 0), 'p3-billion': (0, 14), 'p3-bookcorpus-table': (0, 20)}


def trim(names):
    import os
    from PIL import Image
    from paper_shots import OUT
    for n in names:
        if n in TRIM:
            p = os.path.join(OUT, n + '.png')
            im = Image.open(p)
            left, top = TRIM[n]
            im.crop((left, top, im.width, im.height)).save(p)


if __name__ == '__main__':
    import os
    run(JOBS)
    only = os.environ.get('ONLY')
    trim(only.split(',') if only else [j['name'] for j in JOBS])
