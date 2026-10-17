"""Highlighted excerpts of "Attention Is All You Need" for Part 3 (attention): §3.2, Figure 2, Eq. 1, footnote 4,
§3.2.2 multi-head attention and §3.2.3 the three uses of attention."""
from paper_shots import run

JOBS = [
    # ---------------------------------------------------------------- page 3 (index 2): §3.2 starts
    dict(name='p3-attention-def', page=2, anchor='An attention function can be described',
         highlight=['mapping a query and a set of key-value pairs to an output', 'weighted sum'],
         above=18, below=26),
    # ---------------------------------------------------------------- page 4 (index 3): Figure 2, §3.2.1, footnote 4
    dict(name='p3-figure2', page=3, anchor='Figure 2: (left) Scaled Dot-Product Attention.',
         highlight=['Scaled Dot-Product Attention', 'several attention layers running in parallel'],
         figure=True, column='full', below=0),
    dict(name='p3-weights', page=3, anchor='of the values, where the weight assigned',
         highlight=['compatibility function of the query with the corresponding key'], above=4, end='corresponding key.'),
    dict(name='p3-sdpa', page=3, anchor='We call our particular attention',
         highlight=['queries and keys of dimension dk', 'values of dimension dv', 'dot products of the query with all keys',
                    'apply a softmax function to obtain the weights'],
         above=16, end='We compute the matrix of outputs as'),
    dict(name='p3-eq1', page=3, anchor='We compute the matrix of outputs as', band=(10, 44),
         highlight=[], above=2, below=52),
    dict(name='p3-additive', page=3, anchor='The two most commonly used attention functions',
         highlight=['additive attention [2]', 'except for the scaling factor', 'much faster and more space-efficient in practice'],
         above=4, end='matrix multiplication code.'),
    dict(name='p3-scale-why', page=3, anchor='While for small values of dk',
         highlight=['additive attention outperforms', 'dot products grow large in magnitude', 'extremely small gradients'],
         above=4, end='we scale the dot products by'),
    dict(name='p3-footnote', page=3, anchor='why the dot products get large',
         highlight=['mean 0 and variance 1', 'has mean 0 and variance dk'], above=4, below=14),
    dict(name='p3-multihead-1', page=3, anchor='Instead of performing a single attention function',
         highlight=['linearly project the queries, keys and values h times', 'perform the attention function in parallel'],
         above=16, end='yielding dv-dimensional'),
    # ---------------------------------------------------------------- page 5 (index 4): §3.2.2 end, §3.2.3
    dict(name='p3-multihead-2', page=4, anchor='output values. These are concatenated',
         highlight=['concatenated and once again projected', 'different representation subspaces at different positions',
                    'averaging inhibits this', 'h = 8 parallel attention layers',
                    'similar to that of single-head attention with full dimensionality'],
         above=4, end='full dimensionality.'),
    dict(name='p3-uses-1', page=4, anchor='The Transformer uses multi-head attention in three different ways',
         highlight=['queries come from the previous decoder layer', 'memory keys and values come from the output of the encoder',
                    'attend over all positions in the input sequence'],
         above=16, end='models such as [38, 2, 9]'),
    dict(name='p3-uses-2', page=4, anchor='The encoder contains self-attention layers',
         highlight=['all of the keys, values and queries come from the same place'], above=4,
         end='Each position in the encoder can attend to all positions in the previous layer of the encoder'),
    dict(name='p3-uses-3', page=4, anchor='Similarly, self-attention layers in the decoder',
         highlight=['up to and including that position', 'prevent leftward information flow', 'auto-regressive property',
                    'masking out (setting to −∞)'],
         above=4, end='See Figure 2.'),
]

if __name__ == '__main__':
    run(JOBS)
