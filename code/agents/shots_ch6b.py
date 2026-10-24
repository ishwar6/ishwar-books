"""Highlighted excerpts for Chapter 6, second half (Sections 6.4 to 6.6): the agreement passage of the LLM-as-a-judge paper
(Zheng et al. 2023), tau-bench's reward and pass^k definitions and its Figure 4 (Yao et al. 2024), SWE-bench's fail-to-pass tests
(Jimenez et al. 2023), and the abstracts of WebArena, OSWorld and Terminal-Bench. Usage: python shots_ch6b.py (ONLY=a,b for a subset)."""
import os
from paper_shots import shot
from shots_ch3 import finish

JOBS = [
    dict(name='ch6b-judge-agreement', arxiv_id='2306.05685', page=6, anchor='We compute agreement on MT-bench data',
         highlight=['between GPT-4 and humans reaches 85%', 'agreement among humans (81%)'], above=4, end='make better judgments.'),
    dict(name='ch6b-tau-reward', arxiv_id='2406.12045', page=3, column='full', anchor='Reward. The reward of a task episode',
         highlight=['whether the final database is identical to the unique ground truth outcome database'], above=4,
         end='contain all necessary information'),
    dict(name='ch6b-tau-passk', arxiv_id='2406.12045', page=3, column='full', anchor='Pass^k metric.',
         highlight=['the chance that all k', 'averaged across tasks'], above=4, end='averaged across tasks.'),
    dict(name='ch6b-tau-figure4', arxiv_id='2406.12045', page=6, figure=True, anchor='Figure 4: pass^k', highlight=[], below=0),
    dict(name='ch6b-swe-f2p', arxiv_id='2310.06770', page=1, anchor='log the associated test results before and after',
         highlight=['fail-to-pass test'], above=4, end='installation or runtime errors.'),
    dict(name='ch6b-webarena-abstract', arxiv_id='2307.13854', page=0, column='full', anchor='With advances in generative AI',
         highlight=['end-to-end task success rate of 14.41%', 'human performance of 78.24%'], above=4, end='development of robust agents'),
    dict(name='ch6b-osworld-abstract', arxiv_id='2404.07972', page=0, column='full', anchor='Autonomous agents that accomplish complex',
         highlight=['over 72.36% of the tasks', 'achieves only 12.24% success'], above=4, end='operational knowledge.'),
    dict(name='ch6b-tbench-abstract', arxiv_id='2601.11868', page=0, column='full', anchor='AI agents may soon become capable',
         highlight=['89 tasks', 'score less than 65%'], above=4, end='at tbench.ai.'),
]

TRIM = {'ch6b-webarena-abstract': 250, 'ch6b-osworld-abstract': 250, 'ch6b-tbench-abstract': 250}

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    done, failed = [], []
    for job in JOBS:
        if only and job['name'] not in only.split(','):
            continue
        try:
            shot(**job)
            done.append(job['name'])
        except (SystemExit, Exception) as e:
            failed.append((job['name'], str(e)[:160]))
    finish(done, TRIM)
    for n, e in failed:
        print('FAILED', n, e)
