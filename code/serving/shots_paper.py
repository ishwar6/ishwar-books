"""Highlighted crops from the papers used in Part 6. Uses the shared cropper in code/agents/paper_shots.py.
Images go to public/img/serving/; the manifest (PDF checksums, pages, anchors) to results/shots_paper.json.
Run: python code/serving/shots_paper.py   (downloads each PDF to ~/.cache/papers once)"""
import hashlib, json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agents'))
import paper_shots as ps
ps.OUT = str(HERE.parents[1] / 'public' / 'img' / 'serving')

JOBS = [
    dict(name='paper-distserve-goodput', arxiv_id='2401.09670', page=1, anchor='needs and maximize per-GPU goodput', above=30,
         highlight=['per-GPU goodput, defined as the', 'request rate that can be served adhering to the SLO',
                    'attainment goal (say, 90%) for each GPU provisioned'], end='lower cost per query.'),
    dict(name='paper-distserve-fig1', arxiv_id='2401.09670', page=0, anchor='Figure 1: Performance when serving an LLM', figure=True, below=0),
    dict(name='paper-distserve-md1', arxiv_id='2401.09670', page=4, anchor='Disaggregation enables the prefill phase to function', above=2,
         highlight=['M/D/1 queue'], end='second corresponds to the queuing delay'),
    dict(name='paper-sarathi-stall', arxiv_id='2403.02310', page=5, anchor='Figure 7: A generation stall occurs', figure=True, below=0),
    dict(name='paper-awq-fig2', arxiv_id='2306.00978', page=2, anchor='Figure 2. We observe that we can find 1%', figure=True, below=0),
    dict(name='paper-awq-table4', arxiv_id='2306.00978', page=6, anchor='Table 4. AWQ improves over round-to-nearest', figure=True, below=0, fig_top=148),
    dict(name='paper-smoothquant-fig4', arxiv_id='2211.10438', page=3, anchor='Figure 4: Magnitude of the input activations', figure=True, below=0),
]
# Figures whose captions run on below the first caption block: explicit crop rectangles (x0, y0, x1, y1 in PDF points).
MANUAL = [
    dict(name='paper-slora-pool', arxiv_id='2311.03285', page=4, rect=(50, 64, 300, 201)),
    dict(name='paper-slora-batch', arxiv_id='2311.03285', page=2, rect=(306, 60, 562, 319)),
    dict(name='paper-punica-sgmv', arxiv_id='2310.18547', page=2, rect=(306, 60, 562, 184)),
    dict(name='paper-sarathi-capacity', arxiv_id='2403.02310', page=10, rect=(50, 40, 300, 372)),
]


def manual(name, arxiv_id, page, rect, dpi=190):
    import pymupdf as fitz
    pg = ps.pdf(arxiv_id)[page]
    pg.get_pixmap(clip=fitz.Rect(*rect), dpi=dpi).save(f'{ps.OUT}/{name}.png')
    print(f'{name}: page {page + 1}, rect {rect}')

if __name__ == '__main__':
    ps.run(JOBS)
    for j in MANUAL:
        manual(**j)
    man = []
    for j in JOBS + MANUAL:
        pdf = Path(ps.CACHE) / f'{j["arxiv_id"]}.pdf'
        man.append({**j, 'url': f'https://arxiv.org/pdf/{j["arxiv_id"]}', 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()})
    (HERE / 'results/shots_paper.json').write_text(json.dumps(man, indent=2) + '\n')
