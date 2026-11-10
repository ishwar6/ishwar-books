"""Highlighted paper crops for Part 7 (running DeepSeek). Run: python shots_part7.py  (or ONLY=name1,name2)."""
import os
from paper_shots import shot

V3, ISCA, V4, V41 = '2412.19437', '2505.09343', '2606.19348', '2609.19969'

JOBS = [
    dict(name='p7-v3-deploy', arxiv_id=V3, page=17, anchor='3.4. Inference and Deployment',
         highlight=['separates the prefilling and decoding stages', 'deploy DeepSeek-V3 on the H800 cluster'],
         above=4, below=70, snap=False),
    dict(name='p7-v3-prefill', arxiv_id=V3, page=18, anchor='3.4.1. Prefilling',
         highlight=['4 nodes with 32 GPUs', '32-way Expert Parallelism (EP32)',
                    'duplicates high-load experts and deploys them', 'every 10 minutes',
                    'it will also host one additional redundant expert'],
         above=4, below=284),
    dict(name='p7-v3-decode', arxiv_id=V3, page=18, anchor='3.4.2. Decoding',
         highlight=['40 nodes with 320 GPUs', 'EP320', 'each GPU hosts only one expert',
                    'direct point-to-point transfers over IB'],
         above=4, below=135),
    dict(name='p7-v3-mtp', arxiv_id=V3, page=34, anchor='5.4.3. Multi-Token Prediction Evaluation',
         highlight=['between 85% and 90%', 'delivering 1.8 times', 'TPS (Tokens Per Second)'],
         above=4, below=120),
    dict(name='p7-isca-kv', arxiv_id=ISCA, page=3, anchor='Table 1: KV cache size comparison',
         highlight=['70.272 KB', '327.680 KB', '516.096 KB'], above=4, below=80),
    dict(name='p7-isca-personal', arxiv_id=ISCA, page=3, anchor='Advantages for Personal Use and On-Premises',
         highlight=['activates just 21B parameters', 'nearly 20 tokens per second (TPS)',
                    'single-digit TPS', 'costing approximately $10,000', 'still achieving nearly 20 TPS'],
         above=4, below=154, snap=False),
    dict(name='p7-isca-limit', arxiv_id=ISCA, page=4, anchor='Total Time Per Layer',
         highlight=['14.76 ms TPOT, equivalent to 67 tokens per second'], above=150, below=105),
    dict(name='p7-v4-megamoe', arxiv_id=V4, page=15, anchor='Performance and Open-Sourced Mega-Kernel',
         highlight=['both NVIDIA GPUs and HUAWEI Ascend NPUs', '1.50 ∼1.73× speedup', 'up to 1.96×',
                    'as a component of DeepGEMM'], above=4, below=85),
    dict(name='p7-v4-ondisk', arxiv_id=V4, page=22, anchor='3.5.2. On-Disk KV Cache Storage',
         highlight=['on-disk KV cache storage mechanism to eliminate', 'repeated prefilling for shared-prefix requests',
                    'approximately 8 times larger'], above=4, below=212),
    dict(name='p7-v41-persistent', arxiv_id=V41, page=18, anchor='At the deployment level, we adopt',
         highlight=['Encoder–Prefill–Decode (EPD) disaggregation', 'reduces the persistent KV cache footprint to 1/8',
                    'remain resident for over 72 hours'], above=4, below=260),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            shot(**job)
