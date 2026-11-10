"""Part 5 paper screenshots: NSA (arXiv 2502.11089) and DeepSeek-V3.2 (arXiv 2512.02556).
Run: python shots_part5.py   (or ONLY=name1,name2 python shots_part5.py)"""
import os
from paper_shots import shot

NSA, V32 = '2502.11089', '2512.02556'
JOBS = [
    # ---- NSA ----
    dict(name='p5-nsa-decode-memory', arxiv_id=NSA, page=4, anchor='Specifically for causal self-attention mechanism',
         highlight=['auto-regressive de-', 'coding becomes memory-bandwidth constrained',
                    'requiring loading the entire key-value cache'], above=6, below=66),
    dict(name='p5-nsa-myth', arxiv_id=NSA, page=3, anchor='Performance Degradation: Applying sparsity',
         highlight=['top 20% attention can only cover 70% of the total attention scores'], above=30, below=40),
    dict(name='p5-nsa-figure2', arxiv_id=NSA, page=2, anchor='Figure 2 | Overview of', figure=True,
         highlight=['three parallel attention branches'], below=0),
    dict(name='p5-nsa-framework', arxiv_id=NSA, page=5, anchor='As illustrated in Figure 2, NSA have three mapping strategies',
         highlight=['compression, selection, and sliding window', 'derived from input features via an MLP and sigmoid activation'],
         above=70, below=14),
    dict(name='p5-nsa-compression', arxiv_id=NSA, page=5, anchor='3.3.1. Token Compression',
         highlight=['is a learnable', 'MLP with intra-block position encoding to map keys in a block to a single compressed key'],
         above=4, below=136),
    dict(name='p5-nsa-blockwise', arxiv_id=NSA, page=6, anchor='Blockwise Selection.',
         highlight=['significantly', 'higher throughput for continuous block accesses compared to random index-based reads'],
         above=4, below=80),
    dict(name='p5-nsa-sliding', arxiv_id=NSA, page=7, anchor='3.3.3. Sliding Window',
         highlight=['without being shortcutted by local patterns'], above=4, below=96),
    dict(name='p5-nsa-hparams', arxiv_id=NSA, page=9, anchor='For NSA, we set compression block size',
         highlight=['compression block size', 'selected block count', 'sliding window size'], above=4, below=30),
    dict(name='p5-nsa-speed', arxiv_id=NSA, anchor='up to 9.0× forward and 6.0× backward speedup at 64k context-length',
         highlight=['up to 9.0× forward and 6.0× backward speedup at 64k context-length'], above=50, below=20),
    dict(name='p5-nsa-table4', arxiv_id=NSA, anchor='Table 4 | Memory access volume',
         highlight=['expected speedup is approximately linear with the volume of memory access'],
         above=100, below=34),
    # ---- DeepSeek-V3.2 ----
    dict(name='p5-v32-same-arch', arxiv_id=V32, page=2, anchor='DeepSeek-V3.2 uses exactly the same architecture',
         highlight=['the only architectural modification', 'is the introduction of DeepSeek Sparse Attention (DSA) through continued',
                    'training'], hl_nth={'training': 0}, above=4, below=44),
    dict(name='p5-v32-indexer', arxiv_id=V32, page=2, anchor='The lightning indexer computes the index score',
         highlight=['We choose ReLU as the activation function for throughput consideration',
                    'can be implemented in FP8'], above=4, below=150),
    dict(name='p5-v32-topk', arxiv_id=V32, page=2, anchor='Given the index scores',
         highlight=['retrieves only the key-value entries', 'corresponding to the top-k index scores'], above=4, below=50),
    dict(name='p5-v32-mqa-mode', arxiv_id=V32, page=2, anchor='Instantiate DSA Under MLA',
         highlight=['each key-value entry must be shared across multiple queries for computational',
                    'where each latent vector (the key-value entry of MLA) will be shared across'],
         above=4, below=80),
    dict(name='p5-v32-figure2', arxiv_id=V32, page=3, anchor='Figure 2 | Attention architecture of DeepSeek-V3.2', figure=True,
         highlight=['The', 'green part illustrates how DSA selects the top-k key-value entries according to the indexer'],
         hl_nth={'The': 1}, below=22),
    dict(name='p5-v32-warmup', arxiv_id=V32, page=3, anchor='Dense Warm-up Stage.',
         highlight=['freeze all model parameters except for the lightning', 'resulting in a total of 2.1B tokens'],
         above=4, below=150),
    dict(name='p5-v32-sparse-train', arxiv_id=V32, page=3, anchor='Sparse Training Stage.',
         highlight=['we detach the indexer input from the computational graph',
                    'select 2048 key-value tokens for each query token', 'resulting in a total of 943.7B tokens'],
         above=4, below=185),
    dict(name='p5-v32-parity', arxiv_id=V32, page=4, anchor='2.2. Parity Evaluation',
         highlight=['we do not observe substantial performance degradation'], above=4, below=96),
    dict(name='p5-v32-costs', arxiv_id=V32, page=4, anchor='2.3. Inference Costs',
         highlight=['still has a complexity', 'estimated from', 'benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU',
                    'masked MHA mode to', 'simulate DSA'], above=4, below=140),
    dict(name='p5-v32-figure3', arxiv_id=V32, page=5, anchor='Figure 3 | Inference costs of DeepSeek-V3.1-Terminus', figure=True,
         below=0),
    dict(name='p5-v32-mla-modes', arxiv_id=V32, page=19, anchor='Figure 7 | Illustration of the MHA and MQA modes', figure=True,
         highlight=['MHA mode is used for training and prefilling, while the MQA mode is used for decoding'], below=4),
    dict(name='p5-v32-speciale', arxiv_id=V32, page=5, anchor='DeepSeek-V3.2 and DeepSeek-V3.2-Speciale',
         highlight=['trained exclusively', 'on reasoning data with a reduced length penalty during RL'], above=4, below=84),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            shot(**job)
