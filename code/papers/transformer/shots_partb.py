"""Highlighted excerpts of the papers before the Transformer, for the Background page:
Sutskever et al. 2014 (seq2seq), Bahdanau et al. 2015 (attention), Luong et al. 2015, Wu et al. 2016 (GNMT),
Gehring et al. 2017 (ConvS2S)."""
from paper_shots import run

S2S, BAH, LUO, GNMT, CONV = '1409.3215', '1409.0473', '1508.04025', '1609.08144', '1705.03122'

JOBS = [
    # ---------------------------------------------------------------- Sutskever et al. 2014
    dict(name='pb-s2s-abstract', arxiv_id=S2S, page=0, anchor='In this paper, we present a general end-to-end approach',
         highlight=['multilayered Long Short-Term Memory (LSTM) to map the input sequence',
                    'to a vector of a ﬁxed dimensionality', 'another deep LSTM to decode the', 'target sequence from the vector'],
         above=-1, end='target sequence from the vector.'),
    dict(name='pb-s2s-figure1', arxiv_id=S2S, page=1, anchor='Figure 1: Our model reads an input sentence',
         highlight=['reads the input sentence in reverse'], figure=True, column='full', below=0),
    # ---------------------------------------------------------------- Bahdanau et al. 2015
    dict(name='pb-bah-abstract', arxiv_id=BAH, page=0, anchor='a source sentence into a ﬁxed-length vector from which a decoder',
         highlight=['use of a ﬁxed-length vector is a', 'bottleneck', '(soft-)search', 'for parts of a source sentence that are relevant to predicting a target word'],
         above=13, end='having to form these parts as a hard segment explicitly.'),
    dict(name='pb-bah-issue', arxiv_id=BAH, page=0, anchor='A potential issue with this encoder',
         highlight=['compress all the necessary information of a source sentence into a ﬁxed-length vector',
                    'deteriorates rapidly as the length of an input sentence increases'],
         above=4, end='as the length of an input sentence increases.'),
    dict(name='pb-bah-rnn', arxiv_id=BAH, page=1, anchor='In the Encoder–Decoder framework, an encoder reads',
         highlight=['ht = f (xt, ht−1)', 'hidden state at time t', 'used an LSTM as f'],
         above=4, end='used an LSTM as f and'),
    dict(name='pb-bah-decoder', arxiv_id=BAH, page=2, anchor='The decoder is often trained to predict the next word',
         highlight=['predict the next word', 'ordered conditionals'], above=4, end='the hidden state of the RNN.'),
    dict(name='pb-bah-context', arxiv_id=BAH, page=2, anchor='The context vector ci is, then, computed as a weighted sum',
         highlight=['weighted sum of these', 'alignment model which scores how well the inputs around position j and the output at position',
                    'i match'],
         above=4, end='j-th annotation hj of the input sentence.', column='full'),
    dict(name='pb-bah-attention', arxiv_id=BAH, page=3, anchor='The probability αij, or its associated energy',
         highlight=['this implements a mechanism of attention in the decoder', 'relieve the',
                    'encoder from the burden of having to encode all information'],
         above=4, end='selectively retrieved by the decoder accordingly.'),
    dict(name='pb-bah-figure2', arxiv_id=BAH, page=4, anchor='Figure 2: The BLEU scores',
         highlight=['with respect', 'to the lengths of the sen'], figure=True, below=2),
    dict(name='pb-bah-figure3', arxiv_id=BAH, page=5, anchor='Four sample alignments found by RNNsearch-50',
         highlight=['Each pixel shows the weight αij'], figure=True, column='full', below=0),
    # ---------------------------------------------------------------- Luong et al. 2015
    dict(name='pb-luong-score', arxiv_id=LUO, page=2, anchor='The idea of a global attentional model is to con',
         highlight=['consider all the hidden states of the encoder'], above=4, below=210),
    dict(name='pb-luong-score2', arxiv_id=LUO, page=2, anchor='Here, score is referred as a content-based function',
         highlight=['content-based function', 'dot'], hl_nth={'dot': 0}, above=4, below=80),
    # ---------------------------------------------------------------- Wu et al. 2016, GNMT
    dict(name='pb-gnmt-time', arxiv_id=GNMT, page=14, anchor='it takes around 6 days to train a basic model',
         highlight=['it takes around 6 days to train a basic model using 96 NVIDIA K80 GPUs'], above=30, below=6),
    # ---------------------------------------------------------------- Gehring et al. 2017, ConvS2S
    dict(name='pb-conv-abstract', arxiv_id=CONV, page=0, anchor='Compared to recurrent models',
         highlight=['computations over all', 'elements can be fully parallelized during training',
                    'number of non-linearities', 'is ﬁxed and independent of the input length'],
         above=70, end='tude faster speed'),
]

TRIM = {'pb-s2s-abstract': (300, 14, 12), 'pb-bah-abstract': (300, 0, 0)}   # (left, top, bottom) px: arXiv's side stamp, cut lines

if __name__ == '__main__':
    import os
    from PIL import Image
    from paper_shots import OUT
    run(JOBS)
    for name, (l, t, b) in TRIM.items():
        f = os.path.join(OUT, name + '.png')
        if os.path.exists(f):
            im = Image.open(f)
            if im.width > 1200:                # trim once only
                im.crop((l, t, im.width, im.height - b)).save(f)
