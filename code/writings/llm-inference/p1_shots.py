"""Highlighted paper excerpts for Part 1 of "LLM Inference from the Ground Up" (writings/llm-inference-1-prefill-and-decode).
Uses the shared cutter in code/papers/bert/paper_shots.py; writes PNGs to public/img/writings/llm-inference/.
Run: python p1_shots.py   (or ONLY=p1-sarathi-phases python p1_shots.py)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers', 'bert'))
import paper_shots
from paper_shots import shot
paper_shots.OUT = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'writings', 'llm-inference')

SARATHI = '2403.02310'
JOBS = [
    dict(name='p1-sarathi-phases', arxiv_id=SARATHI, page=1, anchor='The prefill phase is compute-bound because',
         highlight=['The prefill phase is compute-bound', 'the decode phase is memory-bound',
                    'decodes benefit significantly from batching'], above=30,
         end='larger batches can use GPUs more efficiently'),
    dict(name='p1-sarathi-fig3', arxiv_id=SARATHI, page=4, anchor='Figure 3: Throughput of the prefill and decode phases',
         figure=True, below=0, highlight=['prefills are much more efficient than decode', 'batching boosts decode throughput almost linearly']),
]

if __name__ == '__main__':
    only = set(filter(None, os.environ.get('ONLY', '').split(',')))
    for j in JOBS:
        if not only or j['name'] in only:
            print(shot(**j))
