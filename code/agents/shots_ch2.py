"""Highlighted excerpts for Chapter 2: Gorilla, Toolformer (Figure 1), ToolLLM, Generative Agents (architecture figure and
the retrieval score), MemGPT (abstract and Figure 1), Lost in the Middle (Figure 1), and the "Configuring instructions"
page of OpenAI's practical guide (a local PDF)."""
import os
import paper_shots
from paper_shots import run, shot
import pymupdf as fitz

JOBS = [
    dict(name='ch2-gorilla-abstract', arxiv_id='2305.15334', page=0, column='full',
         anchor='Large Language Models (LLMs) have seen an impressive wave of advances',
         highlight=['their potential to effectively use tools via API calls remains unfulfilled',
                    'inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call',
                    'adapt to test-time document changes'],
         above=4, end='TorchHub, and TensorHub APIs.'),
    dict(name='ch2-toolformer-figure1', arxiv_id='2302.04761', page=0, figure=True, column='full',
         anchor='Figure 1: Exemplary predictions of Toolformer', highlight=['autonomously decides to call different APIs'], below=0),
    dict(name='ch2-toolllm-figure1', arxiv_id='2307.16789', page=1, figure=True, column='full',
         anchor='Figure 1: Three phases of constructing ToolBench', highlight=['the API retriever recommends relevant APIs to ToolLLaMA'], below=0),
    dict(name='ch2-toolllm-apis', arxiv_id='2307.16789', page=1, column='full',
         anchor='we gather 16,464 representational state transfer (REST) APIs from RapidAPI',
         highlight=['16,464 representational state transfer (REST) APIs', '49 diverse categories'],
         above=4, below=30),
    dict(name='ch2-genagents-figure5', arxiv_id='2304.03442', page=7, figure=True, column='full',
         anchor='Figure 5: Our generative agent architecture', highlight=['memory stream'], below=0),
    dict(name='ch2-genagents-retrieval', arxiv_id='2304.03442', page=8,
         anchor='To calculate the final retrieval score, we normalize the recency',
         highlight=['weighted combination of the three elements', 'top-ranked memories that fit within the language'],
         above=4, end='are included in the prompt.'),
    dict(name='ch2-memgpt-abstract', arxiv_id='2310.08560', page=0,
         anchor='we propose virtual context management, a technique drawing inspiration',
         highlight=['virtual context management', 'paging between physical memory and disk', 'intelligently manages different storage tiers'],
         above=4, end='within the LLM’s limited context window.'),
    dict(name='ch2-memgpt-figure1', arxiv_id='2310.08560', page=1, figure=True, column='full',
         anchor='Figure 1. MemGPT (left) writes data to persistent memory', highlight=['persistent memory'], below=0),
    dict(name='ch2-litm-figure1', arxiv_id='2307.03172', page=0, figure=True,
         anchor='Figure 1: Changing the location of relevant information', highlight=['U-shaped performance curve'], below=0),
]

OPENAI_PDF = os.path.expanduser('~/.cache/papers/openai-agents-guide.pdf')


def openai_guide():
    orig = paper_shots.pdf
    paper_shots.pdf = lambda _id: fitz.open(OPENAI_PDF)
    try:
        shot('ch2-openai-instructions', arxiv_id='openai-guide', page=10, column='full',
             anchor='High-quality instructions are essential for any LLM-powered app',
             highlight=['Clear instructions reduce ambiguity and improve agent decision-making', 'Use existing documents',
                        'Prompt agents to break down tasks', 'Define clear actions', 'Capture edge cases'],
             above=30, end='if a required piece of info is missing.')
    finally:
        paper_shots.pdf = orig


def finish(names):
    """Trim the rotated arXiv stamp off the left edge where it shows and keep every picture at most 1100 px wide."""
    from PIL import Image
    trim = {'ch2-gorilla-abstract': 70, 'ch2-toolformer-figure1': 70}      # only the crops where the stamp shows
    for f in names:
        f = f + '.png'
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
    only = os.environ.get('ONLY')
    names = [j['name'] for j in JOBS] + ['ch2-openai-instructions']
    names = [n for n in names if not only or n in only.split(',')]
    run(JOBS)
    if 'ch2-openai-instructions' in names:
        openai_guide()
    finish(names)                                 # trim and resize only what was rendered in this run
