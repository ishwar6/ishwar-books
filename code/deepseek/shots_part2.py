"""Paper screenshots for Part 2 (the V3 architecture). Pages are 0-based.
Run: python shots_part2.py   (or ONLY=name1,name2 python shots_part2.py)"""
import os
import numpy as np
from PIL import Image
from paper_shots import shot, OUT


def trim_bottom(name):
    """Cut a half-visible line of text at the bottom edge: end the picture at the last blank row band."""
    path = os.path.join(OUT, name + '.png')
    im = Image.open(path).convert('RGB')
    a = np.asarray(im).min(axis=2)
    blank = (a > 235).all(axis=1)                      # rows with no ink and no highlight
    i = len(blank) - 1
    while i > 0 and not blank[i]:
        i -= 1
    if i > 0.5 * len(blank):
        im = im.crop((0, 0, im.width, i + 1)); blank = blank[:i + 1]
    if not blank[0]:                                   # same at the top edge
        j = 0
        while j < len(blank) - 1 and not blank[j]:
            j += 1
        if j < 0.3 * len(blank):
            im = im.crop((0, j, im.width, im.height))
    im.save(path)

MOE, V2, V3 = '2401.06066', '2405.04434', '2412.19437'
JOBS = [
    # DeepSeek-V3: the block and the headline sizes
    dict(name='p2-v3-figure2', arxiv_id=V3, page=6, anchor='Illustration of the basic architecture of DeepSeek-V3',
         figure=True, column='full', below=22, highlight=['we adopt MLA and DeepSeekMoE']),
    dict(name='p2-v3-hyper', arxiv_id=V3, page=21, anchor='We set the number of Transformer layers to 61', above=4, below=180,
         highlight=['We substitute all FFNs except for the first three layers with MoE layers',
                    '1 shared expert and 256 routed experts', '8 experts will be activated for each token',
                    '671B total parameters, of which 37B are activated for each token']),
    # DeepSeekMoE paper
    dict(name='p2-moe-figure2', arxiv_id=MOE, page=4, anchor='Illustration of DeepSeekMoE', figure=True, column='full',
         below=0, highlight=['the number of expert parameters and computational costs remain constant']),
    dict(name='p2-moe-segment', arxiv_id=MOE, page=4, anchor='In pursuit of the goal', above=4, below=92,
         highlight=['segment each expert FFN into', 'increase the number of activated experts to']),
    dict(name='p2-moe-combos', arxiv_id=MOE, page=5, anchor='From a combinatorial perspective', above=4, below=90,
         highlight=['A typical top-2 routing strategy can yield', 'fine-grained routing strategy can yield']),
    dict(name='p2-moe-shared', arxiv_id=MOE, page=5, anchor='With a conventional routing strategy', above=4, below=160,
         highlight=['multiple experts may converge in acquiring', 'each token will be deterministically assigned to these shared experts']),
    dict(name='p2-moe-collapse', arxiv_id=MOE, page=6, anchor='Automatically learned routing strategies', above=4, below=112,
         highlight=['the model always selects only a few experts', 'Expert-Level Balance Loss']),
    # DeepSeek-V3 MoE and routing
    dict(name='p2-v3-moe-eqs', arxiv_id=V3, page=7, anchor='Basic Architecture of DeepSeekMoE', above=4, below=225,
         highlight=['uses finer-grained experts and isolates some experts as shared ones']),
    dict(name='p2-v3-sigmoid', arxiv_id=V3, page=8, anchor='where', above=4, below=110,
         highlight=['uses the sigmoid function', 'applies a normalization among all selected affinity scores']),
    dict(name='p2-v3-auxfree', arxiv_id=V3, page=8, anchor='Auxiliary-Loss-Free Load Balancing', above=4, below=262,
         highlight=['too large an auxiliary loss will impair', 'we introduce a bias term',
                    'Note that the bias term is only used for routing', 'we will decrease the bias term by']),
    dict(name='p2-v3-seqloss', arxiv_id=V3, page=8, anchor='Complementary Sequence-Wise Auxiliary Loss', above=4, below=240,
         highlight=['to prevent extreme imbalance within any single sequence', 'extremely small']),
    dict(name='p2-v3-node', arxiv_id=V3, page=9, anchor='Node-Limited Routing', above=4, below=150,
         highlight=['each token will be sent to at most', 'nearly achieve full computation-communication overlap',
                    'does not drop any tokens during training']),
    dict(name='p2-v3-ib', arxiv_id=V3, page=12, anchor='in our cluster, cross-node GPUs are fully interconnected', above=4, below=110,
         highlight=['NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB', 'we limit each token to be dispatched to at most 4 nodes',
                    'an average of 3.2 experts per node']),
    dict(name='p2-v3-gamma', arxiv_id=V3, page=22, anchor='We leverage pipeline parallelism', above=4, below=76,
         highlight=['uniformly deployed on 64 GPUs belonging to 8 nodes', 'at most 4 nodes', 'bias update speed']),
    dict(name='p2-v3-batchwise', arxiv_id=V3, page=26, anchor='Batch-Wise Load Balance VS', above=4, below=238,
         highlight=['allows experts to better specialize in different domains', 'using a sequence', 'using the auxiliary-loss-free method']),
    # DeepSeek-V2: MLA
    dict(name='p2-v2-mla-cache', arxiv_id=V2, page=6, anchor='Low-Rank Key-Value Joint Compression', above=4, below=182,
         highlight=['MLA only needs to cache', 'be absorbed into', 'we even do not need to compute keys and values out']),
    dict(name='p2-v2-rope', arxiv_id=V2, page=7, anchor='Decoupled Rotary Position Embedding', above=4, column='full', below=152,
         highlight=['RoPE is incompatible with low-rank KV compression', 'we must recompute the keys for all the prefix tokens',
                    'decoupled RoPE strategy']),
    dict(name='p2-v2-table1', arxiv_id=V2, page=8, anchor='Comparison of the KV cache per token among different attention mechanisms',
         figure=True, column='full', below=60, fig_top=100, highlight=['equal to GQA with only 2.25 groups']),
    # DeepSeek-V3: multi-token prediction
    dict(name='p2-v3-mtp-figure', arxiv_id=V3, page=9, anchor='Illustration of our Multi-Token Prediction', figure=True,
         column='full', below=22, highlight=['We keep the complete causal chain']),
    dict(name='p2-v3-mtp-modules', arxiv_id=V3, page=9, anchor='MTP Modules', above=4, below=70,
         highlight=['a shared embedding layer', 'a shared output head', 'a Transformer block', 'a projection matrix']),
    dict(name='p2-v3-mtp-inference', arxiv_id=V3, page=10, anchor='MTP in Inference', above=4, below=54,
         highlight=['directly discard the MTP modules', 'repurpose these MTP modules for speculative decoding']),
    dict(name='p2-v3-mtp-accept', arxiv_id=V3, page=34, anchor='Multi-Token Prediction Evaluation', above=4, below=125,
         highlight=['ranges between 85% and 90%', 'delivering 1.8 times']),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            shot(**job)
            trim_bottom(job['name'])
