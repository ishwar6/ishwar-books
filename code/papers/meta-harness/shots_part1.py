"""Highlighted excerpts of the paper for Part 1 (title, Figure 1, abstract, §1), plus the source of the 6x claim.
Run: python shots_part1.py   (or ONLY=p1-title,p1-figure1 python shots_part1.py)"""
import os
from PIL import Image
from paper_shots import run, shot, OUT

SWEBM = '2602.09540'   # Tian et al. (2026), SWE-Bench Mobile: reference [47] of the paper, the source of "6x"

JOBS = [
    dict(name='p1-title', page=0, anchor='Meta-Harness: End-to-End Optimization of Model Harnesses',
         highlight=['End-to-End Optimization of Model Harnesses'], above=12, below=4, column='full'),
    dict(name='p1-figure1', page=0, anchor='Figure 1: (Left) On text classification', figure=True, below=0, fig_top=127,
         highlight=['after just 4 evaluations', 'outperforms all reported']),
    dict(name='p1-abstract-1', page=0, anchor='The performance of large language model (LLM) systems depends not',
         highlight=['not only on model weights, but also on their harness', 'what information to store, retrieve, and present to the model'],
         above=4, end='present to the model.'),
    dict(name='p1-abstract-2', page=0, anchor='are still designed largely by hand',
         highlight=['designed largely by hand', 'compress feedback too aggressively', 'they are memoryless, condition only on scalar scores',
                    'short templates or summaries'], above=4, end='short templates or summaries.'),
    dict(name='p1-abstract-3', page=0, anchor='We introduce Meta-Harness, an outer-loop',
         highlight=['outer-loop', 'searches over harness code', 'agentic proposer', 'source code, scores, and execution traces',
                    'through a filesystem'], above=4, end='through a filesystem.'),
    dict(name='p1-abstract-4', page=0, anchor='On online text classification, Meta-Harness improves',
         highlight=['by 7.7 points while using 4× fewer context tokens', 'by 4.7 points on average across five held-out models',
                    'surpass the best hand-engineered', 'richer', 'access to prior experience'],
         above=4, end='automated harness engineering.'),
    dict(name='p1-intro-6x', page=0, anchor='Changing the harness around a fixed large language model',
         highlight=['6×', 'performance gap on the same benchmark [47]', 'often matters as much as the model itself',
                    'harness engineering'], above=4, end='practitioners inspect failures,'),
    dict(name='p1-swebm-6x', arxiv_id=SWEBM, page=4, anchor='Notably, we find that the choice of agent matters',
         highlight=['achieves 12% on Cursor but only 2% on Open-', '6× difference'], above=4, end='a 6× difference.'),
    dict(name='p1-intro-manual', page=1, anchor='adjust heuristics, and iterate on a small number of designs',
         highlight=['adjust heuristics, and iterate on a small number of designs', 'whether', 'this process itself can be automated'],
         above=4, end='this process itself can be automated.'),
    dict(name='p1-intro-textopt', page=1, anchor='A natural starting point is recent work on text optimization',
         highlight=['condition', 'only on the current candidate', 'rely primarily on scalar scores',
                    'short templates or LLM-generated summaries', 'pragmatic scalability choice'],
         above=4, end='dependencies are uninformative.'),
    dict(name='p1-intro-horizon', page=1, anchor='Harnesses act over long horizons',
         highlight=['affect behavior many reasoning steps later', 'trace downstream failures to earlier harness decisions',
                    'ranges from only 100 to 30,000 tokens', 'accessed adaptively'],
         above=4, end='monolithically packed into a single prompt'),
    dict(name='p1-intro-agent', page=1, anchor='We address this limitation with Meta-Harness',
         highlight=['coding agent', 'the amount of experience quickly exceeds context limits',
                    'expose full history through a filesystem', 'grep and cat'],
         above=4, end='rather than ingesting them as a single prompt.'),
    dict(name='p1-intro-82', page=1, anchor='In practice, the proposer reads a median of 82 files',
         highlight=['median of 82 files per iteration', 'over 20 prior candidates per step'], above=4,
         end='per step (Appendix A).'),
    dict(name='p1-intro-10m', page=2, anchor='study, a single evaluation can produce up to 10,000,000 tokens',
         highlight=['up to 10,000,000 tokens of diagnostic information', 'roughly three orders of magnitude'],
         above=4, end='optimization settings (Table 1).'),
    dict(name='p1-intro-results', page=2, anchor='We evaluate Meta-Harness on online text classification, mathematical reasoning',
         highlight=['by 7.7 points while using 4× fewer', 'after 60 proposals', 'with only four', '4.7 points',
                    'ranks #1 among all Haiku 4.5 agents'], above=4, end='and ranks #1 among all Haiku 4.5 agents.'),
]

LEFT_EDGE = 98   # page 1 of the paper: the text starts at x = 98 pt; the arXiv side stamp sits left of it


def trim_stamp(job, dpi=190):
    """Cut the arXiv side stamp off the crops of page 1 (the shared tool keeps it inside the crop)."""
    if job.get('page') != 0 or job.get('arxiv_id'):
        return
    clip = shot(**job, dry=True)['clip']
    path = os.path.join(OUT, job['name'] + '.png')
    im = Image.open(path)
    cut = int((LEFT_EDGE - clip[0]) * dpi / 72)
    if cut > 0:
        im.crop((cut, 0, im.width, im.height)).save(path, optimize=True)
        print(f"{job['name']}: trimmed {cut} px of side stamp")


if __name__ == '__main__':
    run(JOBS)
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            trim_stamp(job)
