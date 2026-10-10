"""Highlighted crops from the papers used in Part 7, via the shared cropper in code/agents/paper_shots.py.
Images go to public/img/multigpu/. Page numbers below are 0-based PDF page indices.
Run: python code/multigpu/shots_paper.py   (downloads each PDF to ~/.cache/papers once)"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agents'))
import paper_shots as ps
ps.OUT = str(HERE.parents[1] / 'public' / 'img' / 'multigpu')

MEG, GP, POPE, L3, DS, DIST, SW, MC = ('1909.08053', '1811.06965', '2211.05102', '2407.21783', '2412.19437',
                                       '2401.09670', '2311.18677', '2407.00079')
JOBS = [
    dict(name='paper-megatron-mlp', arxiv_id=MEG, page=3, anchor='We start by detailing the MLP block.', above=2,
         highlight=['will require a synchronization point before the GeLU', 'partition the ﬁrst GEMM in this column parallel',
                    'split the second GEMM along its rows'], end='operator) and a single all-reduce in the backward pass'),
    dict(name='paper-pope-constant', arxiv_id=POPE, page=3, anchor='As we parallelize the computation across more chips', above=2,
         highlight=['the communication latency remains roughly constant independent of the number of chips used'],
         end='communication becomes a bottleneck'),
    dict(name='paper-gpipe-bubble', arxiv_id=GP, page=3, anchor='As illustrated in Figure 2c, partitioning introduces', above=2,
         highlight=['bubble overhead', 'negligible when M ≥4 × K'], end='negligible when M'),
    dict(name='paper-llama3-pp1', arxiv_id=L3, page=50, anchor='When using a BF16 number representation for the model parameters', above=14,
         highlight=['does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs', 'across 16 GPUs on two machines'],
         end='Within each machine, the high NVLink bandwidth'),
    dict(name='paper-llama3-pp2', arxiv_id=L3, page=51, anchor='enables the use of tensor parallelism (Shoeybi et al., 2019)', above=2,
         highlight=['connectivity has lower bandwidth and higher latency, so we use pipeline parallelism',
                    'they are not an issue during inference', 'also increase latency'], end='better throughput-latency trade-off'),
    dict(name='paper-dsv3-links', arxiv_id=DS, page=12, anchor='in our cluster, cross-node GPUs are fully interconnected with IB', above=2,
         highlight=['NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB', 'at most 4 nodes'],
         end='being blocked by subsequently arriving tokens'),
    dict(name='paper-dsv3-prefill', arxiv_id=DS, page=18, anchor='3.4.1. Prefilling', above=2,
         highlight=['4 nodes with 32 GPUs', 'TP4', 'DP8', 'EP32', 'redundant experts', 'every 10 minutes'],
         end='hosts, it will also host one additional redundant expert'),
    dict(name='paper-dsv3-decode', arxiv_id=DS, page=18, anchor='3.4.2. Decoding', above=2,
         highlight=['40 nodes with 320 GPUs', 'EP320', 'each GPU hosts only one expert', 'point-to-point transfers over IB'],
         end='minimize latency and enhance communication efficiency'),
    dict(name='paper-distserve-fig1', arxiv_id=DIST, page=0, anchor='Figure 1: Performance when serving an LLM with 13B', figure=True, below=0),
    dict(name='paper-distserve-goodput', arxiv_id=DIST, page=1, anchor='resources. To see this, Figure 1 illustrates', above=2,
         highlight=['about 1.6 requests per', '5.6 rps for the prefill phase and 10 rps for decoding', '3.3 rps per GPU'],
         end='requirements (§2.1)'),
    dict(name='paper-distserve-kv', arxiv_id=DIST, page=5, anchor='Communication overhead. Transferring KV caches', above=2,
         highlight=['approximately 1.13GB', '90Gbps bandwidth', 'peak', 'bandwidth between A100 GPUs is 600 GB/s'],
         end='tion in the next section'),
    dict(name='paper-splitwise-layerwise', arxiv_id=SW, page=6, anchor='can be generated in the token generation phase', above=2,
         highlight=['Even when using fast InfiniBand links, the transfer overhead for large prompt',
                    'overlap- ping it with the computation in the prompt phase'], end='reduces the transfer overheads'),
    dict(name='paper-splitwise-result', arxiv_id=SW, page=8, anchor='End-to-end impact.', above=2,
         highlight=['Splitwise only incurs 0.8%', 'adds a 16.5% latency to the second token'],
         end='hardly perceivable even in a user-facing inference'),
    dict(name='paper-mooncake-fig1', arxiv_id=MC, page=1, anchor='Figure 1: Mooncake Architecture.', figure=True, below=0),
]

def megatron_fig3():
    """Figure 3 sits under the running header; crop from the panels to the end of the three-line caption."""
    import pymupdf as fitz
    pg = ps.pdf(MEG)[3]
    pg.get_pixmap(clip=fitz.Rect(300, 60, 545, 348), dpi=190).save(str(Path(ps.OUT) / 'paper-megatron-fig3.png'))
    print('paper-megatron-fig3: page 4, panels (a) and (b) + caption')


def gpipe_fig2():
    """GPipe puts the caption above Figure 2; keep the caption and the three panels below it."""
    import pymupdf as fitz
    pg = ps.pdf(GP)[2]
    pg.get_pixmap(clip=fitz.Rect(100, 66, 512, 318), dpi=190).save(str(Path(ps.OUT) / 'paper-gpipe-fig2.png'))
    print('paper-gpipe-fig2: page 3, caption + panels (a) to (c)')


if __name__ == '__main__':
    ps.run(JOBS)
    gpipe_fig2()
    megatron_fig3()
    man = {'jobs': [{k: v for k, v in j.items()} for j in JOBS], 'pdfs': {}}
    for i in sorted({j['arxiv_id'] for j in JOBS}):
        man['pdfs'][i] = hashlib.sha256((Path(ps.CACHE) / f'{i}.pdf').read_bytes()).hexdigest()
    (HERE / 'results/shots_paper.json').write_text(json.dumps(man, indent=2) + '\n')
