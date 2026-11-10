"""Highlighted paper crops for Part 9 (DeepInfer). Run: python shots_part9.py  (or ONLY=name1,name2)."""
import os
from paper_shots import shot

V3, DSPARK, V41 = '2412.19437', '2607.05147', '2609.19969'

JOBS = [
    dict(name='p9-v3-redundant', arxiv_id=V3, page=18, anchor='3.4.1. Prefilling',
         highlight=['4-way Tensor Parallelism (TP4)', 'duplicates high-load experts and deploys them redundantly',
                    'every 10 minutes'],
         above=4, below=231, snap=False),
    dict(name='p9-dspark-alg', arxiv_id=DSPARK, page=7, anchor='Algorithm 1 Hardware-Aware Prefix Scheduler',
         highlight=['sort descending by', 'Current throughput', 'Expected accepts'],
         above=4, below=300, snap=False),
    dict(name='p9-v41-swa', arxiv_id=V41, page=18, anchor='At the deployment level, we adopt',
         highlight=['Encoder–Prefill–Decode (EPD) disaggregation', 'no longer stores SWA KV',
                    'remain resident for over 72 hours', 'distributed memory pool provisioned from 10% of the host DRAM'],
         above=4, below=470, snap=False),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            shot(**job)
