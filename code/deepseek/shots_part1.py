"""Paper screenshots for Part 1 (every model, and what is open). Run: python shots_part1.py  (or ONLY=name1,name2)."""
import os
from paper_shots import shot

JOBS = [
    dict(name='p1-fireflyer', arxiv_id='2408.14158', page=0, anchor='For DL training, we',
         highlight=['deployed the Fire-Flyer 2 with 10,000 PCIe A100 GPUs', 'reducing costs by half'],
         above=30, below=30),
    dict(name='p1-llm-release', arxiv_id='2401.02954', page=2, anchor='Under the guidance of our scaling laws',
         highlight=['release as much information as possible for community reference', '2 trillion tokens'],
         above=4, below=30),
    dict(name='p1-v2-abstract', arxiv_id='2405.04434', page=0, anchor='We present DeepSeek-V2',
         highlight=['236B total parameters, of which 21B', 'are activated for each token',
                    'context length of 128K tokens', 'The model checkpoints are available at'],
         above=4, below=200, x0_min=62),
    dict(name='p1-v2-lite', arxiv_id='2405.04434', page=4, anchor='we also release DeepSeek-V2-Lite',
         highlight=['we also release DeepSeek-V2-Lite', 'for the open-source community'], above=4, below=30),
    dict(name='p1-v3-abstract', arxiv_id='2412.19437', page=0, anchor='We present DeepSeek-V3',
         highlight=['671B total', 'parameters with 37B activated for each token', '14.8 trillion',
                    '2.788M H800 GPU hours', 'The model checkpoints are available at'],
         above=4, below=180, x0_min=62),
    dict(name='p1-v3-cost', arxiv_id='2412.19437', page=4, anchor='our total training costs amount to only',
         highlight=['excluding the costs', 'associated with prior research and ablation experiments'],
         above=30, below=40),
    dict(name='p1-v3-hai-llm', arxiv_id='2412.19437', page=11, anchor='The training of DeepSeek-V3 is supported by the HAI-LLM',
         highlight=['HAI-LLM framework', 'crafted by our engineers from the ground up'], above=20, below=16),
    dict(name='p1-r1-release', arxiv_id='2501.12948', page=1, anchor='To enable broader access to powerful AI',
         highlight=['we have distilled sev', 'made them publicly available', 'We release DeepSeek-R1 series models to the public'],
         above=4, below=92),
    dict(name='p1-r1-data', arxiv_id='2501.12948', page=6, anchor='In total, we curated 66,000 data pairs',
         highlight=['publicly available', 'open-source datasets or from users who have explicitly consented to share their data'],
         above=4, below=40),
    dict(name='p1-v4-abstract', arxiv_id='2606.19348', page=0, anchor='We present a preview version of DeepSeek-V4',
         highlight=['1.6T parameters (49B activated)', '284B parameters (13B activated)', 'one million tokens',
                    'more than 32T', 'The model checkpoints'],
         above=4, below=245, x0_min=62),
    dict(name='p1-v4-megamoe', arxiv_id='2606.19348', page=15, anchor='Performance and Open-Sourced Mega-Kernel',
         highlight=['We have open-sourced the CUDA-based mega-kernel implementation named', 'as a component of DeepGEMM'], above=4, below=74),
    dict(name='p1-v41-abstract', arxiv_id='2609.19969', page=0, anchor='we introduce DeepSeek-V4.1-Flash',
         highlight=['552B backbone parameters', 'one million tokens',
                    'activates 16B parameters', 'per token during decode but only 8B parameters during prefill', '45T tokens'],
         above=4, below=150, x0_min=62),
    dict(name='p1-v3-longtermism', arxiv_id='2412.19437', page=34, anchor='DeepSeek consistently adheres',
         highlight=['DeepSeek consistently adheres to the route of open-source models with longtermism'], above=4, below=22),
]

# (top, bottom) pixels to cut: the 4 pt snap margin can show a sliver of the next line; the bold Fire-Flyer paper needs both
TRIM = {'p1-fireflyer': (9, 9)}
DEFAULT_TRIM = (0, 6)


def trim(name, cut):
    from PIL import Image
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'public', 'img', 'deepseek', 'papers', name + '.png')
    im = Image.open(path)
    im.crop((0, cut[0], im.width, im.height - cut[1])).save(path)


if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in JOBS:
        if not only or job['name'] in only.split(','):
            shot(**job)
            trim(job['name'], TRIM.get(job['name'], DEFAULT_TRIM))
