"""Highlighted excerpts for Chapter 1: ReAct, Toolformer, Reflexion, Generative Agents, the MAST failure taxonomy,
and the "When should you build an agent?" page of OpenAI's practical guide (a local PDF, not an arXiv paper)."""
import os
import paper_shots
from paper_shots import run, shot
import pymupdf as fitz

JOBS = [
    dict(name='ch1-react-abstract', arxiv_id='2210.03629', page=0,
         anchor='In this paper, we explore the use of LLMs to generate both reasoning traces',
         highlight=['generate both reasoning traces and task-specific actions in an interleaved manner',
                    'reasoning traces help the model induce, track, and update action plans',
                    'actions allow it to interface with and gather additional information from external sources'],
         column='full', above=4, end='such as knowledge bases or environments.'),
    dict(name='ch1-react-figure1', arxiv_id='2210.03629', page=1, figure=True, column='full',
         anchor='Figure 1: (1) Comparison of 4 prompting methods', highlight=['(d) ReAct (Reason+Act)'], below=0),
    dict(name='ch1-toolformer-abstract', arxiv_id='2302.04761', page=0,
         anchor='In this paper, we show that LMs can teach themselves',
         highlight=['LMs can teach themselves to use external tools via simple APIs',
                    'decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results'],
         above=4, end='demonstrations for each API.'),
    dict(name='ch1-reflexion-abstract', arxiv_id='2303.11366', page=0,
         anchor='We propose Reflexion, a novel framework',
         highlight=['not by updating weights, but instead through linguistic feedback', 'verbally reflect on task feedback signals',
                    'episodic memory buffer'],
         column='full', above=4, end='in subsequent trials.'),
    dict(name='ch1-genagents-figure1', arxiv_id='2304.03442', page=0, figure=True, column='full',
         anchor='Figure 1: Generative agents are believable simulacra of human behavior',
         highlight=['populating a sandbox environment, reminiscent of The Sims, with twenty-five agents'], below=0),
    dict(name='ch1-mast-abstract', arxiv_id='2503.13657', page=0,
         anchor='We introduce MAST-Data, a comprehensive dataset',
         highlight=['1600+ annotated traces collected across 7 popular MAS frameworks', '14 unique modes, clustered into 3 categories',
                    'system design issues', 'inter-agent misalignment', 'task verification'],
         column='full', above=4, end='(iii) task verification.'),
    dict(name='ch1-mast-figure1', arxiv_id='2503.13657', page=1, figure=True, column='full',
         anchor='Figure 1: MAST: A Taxonomy of MAS Failure Modes', highlight=['A Taxonomy of MAS Failure Modes'], below=0),
]

OPENAI_PDF = os.path.expanduser('~/.cache/papers/openai-agents-guide.pdf')


def openai_guide():
    """paper_shots downloads arXiv ids; the OpenAI guide is a local PDF, so swap the loader for this one job."""
    orig = paper_shots.pdf
    paper_shots.pdf = lambda _id: fitz.open(OPENAI_PDF)
    try:
        shot('ch1-openai-when-to-build', arxiv_id='openai-guide', page=5, column='full',
             anchor='prioritize workflows that have previously resisted automation',
             highlight=['Complex decision-making', 'Difficult-to-maintain rules', 'Heavy reliance on unstructured data',
                        'a deterministic solution may suffice'],
             above=4, end='a deterministic solution may suffice.')
    finally:
        paper_shots.pdf = orig




def finish():
    """Trim the rotated arXiv stamp off the left edge of the one-column crops and keep every picture at most 1100 px wide."""
    from PIL import Image
    trim = {'ch1-react-abstract': 70, 'ch1-reflexion-abstract': 70, 'ch1-mast-abstract': 60, 'ch1-genagents-figure1': 78}
    for f in sorted(os.listdir(paper_shots.OUT)):
        if not f.startswith('ch1-') or not f.endswith('.png'):
            continue
        path = os.path.join(paper_shots.OUT, f)
        im = Image.open(path)
        t = trim.get(f[:-4], 0)
        if t:
            im = im.crop((t, 0, im.width, im.height))
        if im.width > 1100:
            im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
        im.save(path, optimize=True)
        print(f'{f}: {im.width}x{im.height}')


if __name__ == '__main__':
    run(JOBS)
    if not os.environ.get('ONLY') or 'ch1-openai-when-to-build' in os.environ.get('ONLY', ''):
        openai_guide()
    finish()
