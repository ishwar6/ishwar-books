"""Paper screenshots for Part 3 (training DeepSeek-V3). Run: python shots_part3.py  (or ONLY=name1,name2)."""
import os
from paper_shots import shot

V3, ISCA, FF, V4 = '2412.19437', '2505.09343', '2408.14158', '2606.19348'
JOBS = [
    dict(name='p3-stable', arxiv_id=V3, page=3, anchor='During pre-training, we train DeepSeek-V3 on 14.8T',
         highlight=['14.8T high-quality and diverse tokens', 'did not', 'encounter any irrecoverable loss spikes or have to roll back'],
         above=6, below=40),
    dict(name='p3-cost', arxiv_id=V3, page=4, anchor='Training Costs', figure=False,
         highlight=['Table 1 | Training costs of DeepSeek-V3, assuming the rental price of H800 is $2 per GPU hour.'],
         above=8, below=40, column='full'),
    dict(name='p3-cost-text', arxiv_id=V3, page=4, anchor='During the pre-training stage, training DeepSeek-V3 on each',
         highlight=['180K', 'H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs',
                    'excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data'],
         above=40, below=50),
    dict(name='p3-data', arxiv_id=V3, page=20, anchor='Compared with DeepSeek-V2, we optimize the pre-training corpus',
         highlight=['enhancing the ratio', 'of mathematical and programming samples', 'expanding multilingual coverage beyond'],
         above=4, below=30),
    dict(name='p3-data2', arxiv_id=V3, page=21, anchor='English and Chinese. Also, our data processing pipeline',
         highlight=['do not incorporate cross-sample attention masking', '14.8T high-quality and diverse',
                    'The FIM', 'strategy is applied at a rate of 0.1', 'Byte-level BPE', 'extended', 'vocabulary of 128K tokens'],
         above=4, below=300),
    dict(name='p3-lr', arxiv_id=V3, page=21, anchor='Training Hyper-Parameters.',
         highlight=['linearly increase it from 0 to 2.2', 'during the first', '2K steps', 'constant learning rate of 2.2',
                    'until the model consumes 10T', 'decay the learning rate to 2.2', 'in 4.3T tokens',
                    'following a cosine decay curve', 'final 500B tokens'],
         above=4, below=120),
    dict(name='p3-batch', arxiv_id=V3, page=22, anchor='of 7.3', highlight=[
        'batch size is gradually increased from 3072 to 15360', 'in the training of the first 469B tokens',
        'uniformly deployed on 64 GPUs belonging to 8 nodes'], above=4, below=70),
    dict(name='p3-yarn', arxiv_id=V3, page=22, anchor='4.3. Long Context Extension',
         highlight=['two additional training phases, each comprising 1000 steps', 'from 4K to 32K and then to 128K',
                    'length is set to 32K, and the batch size is 1920', 'increased to 128K, and the batch size is reduced to 480'],
         above=4, below=150),
    dict(name='p3-cluster', arxiv_id=V3, page=10, anchor='3.1. Compute Clusters',
         highlight=['2048 NVIDIA H800 GPUs', '8 GPUs connected by NVLink and NVSwitch within nodes', 'InfiniBand (IB) interconnects'],
         above=4, below=62, column='full'),
    dict(name='p3-framework', arxiv_id=V3, page=11, anchor='3.2. Training Framework',
         highlight=['16-way Pipeline Parallelism (PP)', '64-way Expert Paral', 'spanning 8 nodes', 'ZeRO-1 Data Parallelism (DP)',
                    'without using costly Tensor Parallelism (TP)'],
         above=4, below=150, column='full'),
    dict(name='p3-isca-h800', arxiv_id=ISCA, page=6, anchor='The NVIDIA H800 GPU SXM architecture we currently use',
         highlight=['reduced FP64 computational', 'performance and NVLink bandwidth for regulatory compliance',
                    'from 900 GB/s to 400 GB/s', 'eight 400G', 'Infiniband (IB) CX7 NICs', 'Avoidance of Tensor Parallelism (TP)',
                    'inefficiency under limited', 'NVLink bandwidth'],
         above=4, below=240),
    dict(name='p3-fireflyer', arxiv_id=FF, page=0, anchor='For DL training, we deployed the Fire-Flyer 2',
         highlight=['10,000 PCIe A100 GPUs', 'reducing costs by half and energy consumption by 40%'], above=40, below=30),
    dict(name='p3-dualpipe-ratio', arxiv_id=V3, page=11, anchor='3.2.1. DualPipe and Computation-Communication Overlap',
         highlight=['inefficient computation-to-communication ratio of approximately 1:1',
                    'The key idea of DualPipe is to overlap the computation and communication within a pair of',
                    'individual forward and backward chunks', 'attention, all-to-all dispatch, MLP, and all-to-all combine'],
         above=4, below=60, column='full'),
    dict(name='p3-fig4', arxiv_id=V3, page=11, anchor='Figure 4 | Overlapping strategy', figure=True,
         highlight=['Both all-to-all and PP communication can be fully hidden.'], below=0, column='full'),
    dict(name='p3-fig5', arxiv_id=V3, page=12, anchor='Figure 5 | Example DualPipe scheduling', figure=True,
         highlight=['8 PP ranks and 20 micro-batches in two directions'], below=0, column='full'),
    dict(name='p3-table2', arxiv_id=V3, page=12, anchor='Table 2 | Comparison of pipeline bubbles', figure=True,
         highlight=['Comparison of pipeline bubbles and memory usage'], below=0, column='full', fig_top=70),
    dict(name='p3-alltoall', arxiv_id=V3, page=12, anchor='NVLink offers a bandwidth of 160 GB/s',
         highlight=['NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB', '(50 GB/s)',
                    'at most 4 nodes', 'first be transmitted via IB to the GPUs with the same in-node',
                    'index on its target nodes', 'forwarded via NVLink'],
         above=10, below=90, column='full'),
    dict(name='p3-sms', arxiv_id=V3, page=13, anchor='selects only 8 routed experts in practice',
         highlight=['maximum of 13 experts', 'only 20 SMs are sufficient to fully utilize the bandwidths of IB',
                    'and NVLink'], above=4, below=45, column='full'),
    dict(name='p3-memsave', arxiv_id=V3, page=13, anchor='3.2.3. Extremely Memory Saving with Minimal Overhead',
         highlight=['recompute all RMSNorm op', 'MLA up-projections during back-propagation',
                    'stored in CPU memory and are updated', 'asynchronously',
                    'shallowest layers (including the embedding layer) and deepest layers (including',
                    'the output head) of the model on the same PP rank'],
         above=4, below=300, column='full'),
    dict(name='p3-fig6', arxiv_id=V3, page=14, anchor='Figure 6 | The overall mixed precision framework', figure=True,
         highlight=['only the Linear operator is illustrated'], below=0, column='full'),
    dict(name='p3-fp8-025', arxiv_id=V3, page=14, anchor='pre-training (Fishman et al., 2024). To address this challenge',
         highlight=['tile-wise grouping', 'with 1 × 𝑁𝑐elements', 'block-wise grouping with',
                    'relative loss error of our FP8-training model remains', 'consistently below 0.25%'],
         above=4, below=150, column='full'),
    dict(name='p3-fp8-keep', arxiv_id=V3, page=14, anchor='Despite the efficiency advantage of the FP8 format',
         highlight=['the embedding module, the output head, MoE gating modules,', 'normalization operators, and attention operators',
                    'store the master weights, weight gradients, and optimizer states in higher precision'],
         above=4, below=70, column='full'),
    dict(name='p3-fig7', arxiv_id=V3, page=15, anchor='Figure 7 | (a) We propose a fine-grained quantization', figure=True,
         highlight=['promoting to CUDA', 'Cores at an interval of'], below=0, column='full'),
    dict(name='p3-tiles', arxiv_id=V3, page=15, anchor='In low-precision training frameworks, overflows and underflows',
         highlight=['highly sensitive to activation outliers', 'we group and', 'scale elements on a 1x128 tile basis',
                    'per token per 128 channels', 'on a 128x128 block basis'],
         above=20, below=200, column='full'),
    dict(name='p3-accum', arxiv_id=V3, page=16, anchor='Increasing Accumulation Precision.',
         highlight=['limited to', 'retaining around 14 bits', 'maximum', 'relative error of nearly 2%',
                    'copied to FP32 registers on CUDA Cores'],
         above=4, below=200, column='full'),
    dict(name='p3-nc128', arxiv_id=V3, page=16, anchor='on the H800',
         highlight=['setting 𝑁𝐶= 128 elements, equivalent to 4 WGMMAs', 'E4M3 format on all tensors'],
         above=4, below=110, column='full'),
    dict(name='p3-blockwise', arxiv_id=V3, page=46, anchor='B.2. Discussion About Block-Wise Quantization',
         highlight=['block-wise quantization of activation gradients leads to'], above=4, below=150, column='full'),
    dict(name='p3-blockwise2', arxiv_id=V3, page=47, anchor='model divergence on an MoE model',
         highlight=['model divergence on an MoE model comprising approximately 16B total parameters', 'token-correlated outliers'],
         above=4, below=40, column='full'),
    dict(name='p3-isca-flops', arxiv_id=ISCA, page=3, anchor='Table 2: Comparison of computational costs',
         highlight=['250 GFLOPS/Token', 'sequence length of 4096'], above=4, below=85),
    dict(name='p3-v4-instability', arxiv_id=V4, page=25, anchor='4.2.3. Mitigating Training Instability',
         highlight=['We encountered notable instability challenges during training.',
                    'simple rollbacks could temporarily restore the training state',
                    'the occurrence of spikes is consistently tied to outliers in the MoE layers'],
         above=4, below=80, column='full'),
]

def trim_partial_lines(name):
    """Cut a half-shown line of text at the top or bottom edge of a text crop (the crop snaps to lines, but a
    line's box can be shorter than its glyphs, so a sliver of the next line sometimes shows)."""
    from PIL import Image
    import numpy as np
    path = os.path.join(OUT, name + '.png')
    im = Image.open(path).convert('RGB')
    a = np.asarray(im)
    ink = (a.min(axis=2) < 235).any(axis=1)          # a row with any dark or highlighted pixel
    segs, start = [], None
    for y, v in enumerate(ink):
        if v and start is None:
            start = y
        if not v and start is not None:
            segs.append((start, y)); start = None
    if start is not None:
        segs.append((start, len(ink)))
    if len(segs) < 3:
        return
    hs = sorted(e - s for s, e in segs)
    med = hs[len(hs) // 2]
    top, bot = 0, a.shape[0]
    if segs[-1][1] >= a.shape[0] - 1 and segs[-1][1] - segs[-1][0] < 0.75 * med:
        bot = segs[-1][0] - 2
    if segs[0][0] <= 1 and segs[0][1] - segs[0][0] < 0.75 * med:
        top = segs[0][1] + 2
    if (top, bot) != (0, a.shape[0]):
        im.crop((0, top, im.width, bot)).save(path)
        print(f'   {name}: trimmed to rows {top}..{bot}')


OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'public', 'img', 'deepseek', 'papers')

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            shot(**job)
            if not job.get('figure') and job['name'] not in ('p3-cost', 'p3-isca-flops'):
                trim_partial_lines(job['name'])
