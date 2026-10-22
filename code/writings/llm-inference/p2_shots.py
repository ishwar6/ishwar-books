"""Highlighted paper excerpts for Part 2 of "LLM Inference from the Ground Up" (writings/llm-inference-2-kv-cache).
Uses the shared cutter in code/papers/bert/paper_shots.py; writes PNGs to public/img/writings/llm-inference/.
The vLLM paper crops that already exist in public/img/papers/vllm/ are reused as they are.
Run: python p2_shots.py   (or ONLY=p2-mqa-abstract python p2_shots.py)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers', 'bert'))
import paper_shots
from paper_shots import shot
paper_shots.OUT = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'writings', 'llm-inference')

MQA, GQA, DSV2, VLLM = '1911.02150', '2305.13245', '2405.04434', '2309.06180'
JOBS = [
    dict(name='p2-mqa-abstract', arxiv_id=MQA, page=0, anchor='incremental inference (where such',
         highlight=['the memory-bandwidth cost of repeatedly loading the large',
                    'the keys and values are shared across all of the different attention',
                    'only minor quality degradation'],
         above=30, end='incur only minor quality degradation from the baseline'),
    dict(name='p2-gqa-abstract', arxiv_id=GQA, page=0, anchor='introduce grouped-query attention (GQA)',
         highlight=['grouped-query attention (GQA)', 'an intermediate (more than one, less than',
                    'achieves quality close to multi-head attention with comparable speed to MQA'],
         above=40, end='comparable speed to MQA'),
    dict(name='p2-deepseek-kv', arxiv_id=DSV2, page=0, anchor='reduces the KV cache by 93.3%',
         highlight=['Multi-head Latent Attention (MLA)', 'reduces the KV cache by 93.3%'],
         above=60, end='boosts the maximum generation throughput to 5.76 times'),
    dict(name='p2-figure13', arxiv_id=VLLM, page=9, anchor='Figure 13. Average number of batched requests',
         figure=True, below=0, fig_top=100, highlight=['Average number of batched requests']),
]

TRIM = {'p2-mqa-abstract': 190, 'p2-deepseek-kv': 120}

if __name__ == '__main__':
    only = set(filter(None, os.environ.get('ONLY', '').split(',')))
    for j in JOBS:
        if not only or j['name'] in only:
            print(shot(**j))
            if j['name'] in TRIM:                     # cut arXiv's rotated side stamp off the left edge
                from PIL import Image
                path = os.path.join(paper_shots.OUT, j['name'] + '.png')
                im = Image.open(path)
                im.crop((TRIM[j['name']], 0, im.width, im.height)).save(path)
