"""Highlighted excerpts of "Attention Is All You Need" for Part 4 (Why Self-Attention): §4 and Table 1."""
from paper_shots import run

JOBS = [
    dict(name='p4-intro', anchor='4 Why Self-Attention',
         highlight=['recurrent and convolutional layers', 'one variable-length sequence of symbol representations', 'three desiderata'],
         above=4, end='three desiderata.'),
    dict(name='p4-desiderata', anchor='One is the total computational complexity per layer',
         highlight=['total computational complexity per layer', 'minimum number of sequential operations required',
                    'path length between long-range dependencies', 'The shorter these paths', 'compare the maximum path length'],
         above=4, end='composed of the different layer types.'),
    dict(name='p4-table1', anchor='Table 1: Maximum path lengths, per-layer complexity',
         highlight=['n is the sequence length, d is the representation dimension'], above=4, below=112),
    dict(name='p4-constant', anchor='As noted in Table 1, a self-attention layer connects',
         highlight=['constant number of sequentially executed operations', 'requires O(n) sequential operations',
                    'self-attention layers are faster than recurrent layers when the sequence'],
         above=4, below=26),
    dict(name='p4-n-smaller-d', anchor='length n is smaller than the representation dimensionality d',
         highlight=['length n is smaller than the representation dimensionality d', 'word-piece', 'byte-pair'],
         above=4, end='byte-pair [31] representations.'),
    dict(name='p4-restricted', anchor='To improve computational performance for tasks involving',
         highlight=['neighborhood of size r', 'increase the maximum path length to O(n/r)'], above=4, end='further in future work.'),
    dict(name='p4-conv', anchor='A single convolutional layer with kernel width',
         highlight=['does not connect all pairs of input and output', 'stack of O(n/k) convolutional layers', 'dilated convolutions',
                    'more expensive than recurrent layers, by a factor of k', 'Separable convolutions',
                    'equal to the combination of a self-attention layer and a point-wise feed-forward layer'],
         above=4, end='the approach we take in our model.'),
    dict(name='p4-interpret', anchor='As side benefit, self-attention could yield more interpretable',
         highlight=['more interpretable models', 'individual attention', 'syntactic', 'semantic structure'],
         above=4, end='structure of the sentences.'),
]

if __name__ == '__main__':
    run(JOBS)
