"""Highlighted crops from the SGLang paper (arXiv 2312.07104v2) for Part 5.
Uses the shared cropper in code/agents/paper_shots.py; images go to public/img/sglang/.
Run: python code/sglang/shots_paper.py   (downloads the PDF to ~/.cache/papers once)"""
import hashlib, json, os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agents'))
import paper_shots as ps
ps.OUT = str(HERE.parents[1] / 'public' / 'img' / 'sglang')
ID = '2312.07104'

JOBS = [
    dict(name='paper-radix-lru', arxiv_id=ID, page=3, anchor='RadixAttention. A radix tree is a data structure', above=2,
         highlight=['evicts the least recently used leaf first', 'each node maintains a reference counter',
                    'A node is evictable if its reference counter is zero'], end='in favor of a larger batch size'),
    dict(name='paper-schedule', arxiv_id=ID, page=5, anchor='requests by matched prefix length and prioritize', above=2,
         highlight=['prioritize requests with longer matched prefixes', 'depth-first search order',
                    'it can lead to starvation'], end='as future work'),
    dict(name='paper-fig4', arxiv_id=ID, page=5, anchor='Figure 4: The decoding process of normal and compressed FSMs', figure=True, below=0),
    dict(name='paper-fig5', arxiv_id=ID, page=6, anchor='Figure 5: Normalized throughput on Llama-7B models', figure=True, below=8),
    dict(name='paper-arena', arxiv_id=ID, page=7, anchor='Production deployment.', above=2,
         highlight=['52.4% RadixAttention cache hit rate', '74.1% for Vicuna-33B', 'average of 1.7'], end='average of 1.7'),
    dict(name='paper-overhead', arxiv_id=ID, page=8, anchor='Overhead of RadixAttention.', above=2,
         highlight=['only 0.2 seconds'], end='turn on RadixAttention by default'),
    dict(name='paper-alg1', arxiv_id=ID, page=14, anchor='Algorithm 1 Cache-Aware Scheduling', highlight=['requests.sort()'],
         end='return finished_requests', column='full'),
]

def fig3():
    """Figure 3 has a 20-line caption; keep the nine panels and the caption's first two lines."""
    import pymupdf as fitz
    pg = ps.pdf(ID)[4]
    cap = pg.search_for('Figure 3: Examples of RadixAttention operations')[0]
    clip = fitz.Rect(100, 42, 512, cap.y1 + 11.5)
    pg.get_pixmap(clip=clip, dpi=190).save(os.path.join(ps.OUT, 'paper-fig3.png'))
    print('paper-fig3: page 5, nine panels + first caption lines')


if __name__ == '__main__':
    ps.run(JOBS)
    fig3()
    pdf = Path(ps.CACHE) / f'{ID}.pdf'
    man = {'url': 'https://arxiv.org/pdf/2312.07104v2', 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
           'jobs': [{k: v for k, v in j.items() if k != 'arxiv_id'} for j in JOBS]}
    (HERE / 'results/shots_paper.json').write_text(json.dumps(man, indent=2) + '\n')
