"""Highlighted excerpts of "Attention Is All You Need" for Part 7 (build it from scratch): the lines that define each
block we write in code (§3.1, §3.2.1, §3.2.2, §3.2.3, §3.3, §3.4, §3.5) and the training and decoding rules (§5.3,
§5.4, §6.1)."""
from paper_shots import run

JOBS = [
    # ---------------------------------------------------------------- page 5 (index 4): §3.4 embeddings
    dict(name='p7-embed', page=4, anchor='Similarly to other sequence transduction models, we use learned',
         highlight=['share the same weight matrix between the two embedding layers and the pre-softmax',
                    'multiply those weights by'],
         above=16, end='we multiply those weights by'),
    # ---------------------------------------------------------------- page 6 (index 5): §3.5 the formula
    dict(name='p7-pe', page=5, anchor='In this work, we use sine and cosine functions of different frequencies',
         highlight=['sine and cosine functions of different frequencies'], band=(4, 50), above=4, below=58),
    # ---------------------------------------------------------------- page 4 (index 3): Eq. 1
    dict(name='p7-eq1', page=3, anchor='We call our particular attention',
         highlight=['divide each by', 'apply a softmax function to obtain the weights'], above=16, below=118),
    # ---------------------------------------------------------------- page 5 (index 4): multi-head equation
    dict(name='p7-multihead', page=4, anchor='output values. These are concatenated',
         highlight=['concatenated and once again projected', 'h = 8 parallel attention layers', 'dk = dv = dmodel/h = 64'],
         above=4, end='full dimensionality.'),
    dict(name='p7-mask', page=4, anchor='Similarly, self-attention layers in the decoder',
         highlight=['up to and including that position', 'masking out (setting to −∞)'], above=4, end='See Figure 2.'),
    dict(name='p7-ffn', page=4, anchor='In addition to attention sub-layers, each of the layers',
         highlight=['applied to each position separately and identically', 'two linear transformations with a ReLU activation'],
         above=16, end='dff = 2048.'),
    # ---------------------------------------------------------------- page 3 (index 2): §3.1
    dict(name='p7-residual', page=2, anchor='The encoder is composed of a stack of N = 6 identical layers',
         highlight=['residual connection [11] around each of', 'followed by layer normalization',
                    'LayerNorm(x + Sublayer(x))'],
         above=4, end='produce outputs of dimension'),
    dict(name='p7-decoder', page=2, anchor='The decoder is also composed of a stack of N = 6 identical layers',
         highlight=['inserts a third sub-layer', 'multi-head attention over the output of the encoder stack',
                    'output embeddings are offset by one position'],
         above=4, end='known outputs at positions less than i.'),
    # ---------------------------------------------------------------- page 7 (index 6): Eq. 3
    dict(name='p7-eq3', page=6, anchor='We used the Adam optimizer',
         highlight=['β1 = 0.9, β2 = 0.98 and ϵ = 10−9', 'warmup_steps = 4000'], above=16, end='We used warmup_steps = 4000.'),
    # ---------------------------------------------------------------- page 8 (index 7): §5.4 and §6.1
    dict(name='p7-dropout', page=7, anchor='Residual Dropout',
         highlight=['dropout [33] to the output of each sub-layer', 'sums of the embeddings and the', 'Pdrop = 0.1'],
         above=4, end='Pdrop = 0.1.'),
    dict(name='p7-smoothing', page=7, anchor='Label Smoothing',
         highlight=['label smoothing of value ϵls = 0.1', 'the model learns to be more unsure'], above=4,
         end='improves accuracy and BLEU score.'),
    dict(name='p7-beam', page=7, anchor='We used beam search with a beam size of 4',
         highlight=['beam search with a beam size of 4 and length penalty α = 0.6', 'input length + 50'],
         above=4, end='terminate early when possible [38].'),
]

if __name__ == '__main__':
    run(JOBS)
