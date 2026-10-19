"""Highlighted excerpts for the Background page (attention, keys and values, and the causal mask).
From Vaswani et al. (2017), Attention Is All You Need (1706.03762). Run: python shots_partb.py   (or ONLY=pb-name1,pb-name2)"""
from paper_shots import run

TRANSFORMER = '1706.03762'

JOBS = [
    dict(name='pb-attn-def', arxiv_id=TRANSFORMER, page=2, anchor='An attention function can be described as mapping a query',
         highlight=['mapping a query and a set of key-value pairs to an output', 'weighted sum'],
         above=4, end='The output is computed as a weighted sum'),
    dict(name='pb-attn-def-2', arxiv_id=TRANSFORMER, page=3, anchor='of the values, where the weight assigned to each value',
         highlight=['of the values', 'compatibility function of the', 'query with the corresponding key'],
         above=4, end='query with the corresponding key.'),
    dict(name='pb-attn-eq', arxiv_id=TRANSFORMER, page=3, anchor='We call our particular attention',
         highlight=['dot products of the', 'query with all keys, divide each by', 'apply a softmax function to obtain the weights on the',
                    'values'], hl_nth={'values': 2}, above=5, below=120, snap=False),
    dict(name='pb-mask', arxiv_id=TRANSFORMER, page=4, anchor='Similarly, self-attention layers in the decoder allow',
         highlight=['up to and including that position', 'preserve the auto-regressive property', 'masking out (setting to −∞)'],
         above=4, end='illegal connections. See Figure 2.'),
]

if __name__ == '__main__':
    run(JOBS)
