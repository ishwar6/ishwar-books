"""Highlighted excerpts for Chapter 2 (how it started, and how it evolved): the papers from preference learning
(Christiano 2017) to DeepSeek-R1 (2025). Each job crops a figure, an equation or a few sentences out of the arXiv PDF
and highlights the key phrase.  Usage: python shots_ch2.py   (or ONLY=name1,name2 python shots_ch2.py)"""
import os
import paper_shots
from paper_shots import run, pdf, find, YELLOW, OUT
import pymupdf as fitz


def manual(name, arxiv_id, page, rect, highlight=(), dpi=190):
    """A fixed crop rectangle (points) for figures that sit beside body text, which the automatic crop cannot isolate."""
    pg = pdf(arxiv_id)[page]
    for phrase in highlight:
        rects = find(pg, phrase)
        if not rects:
            raise SystemExit(f'{name}: highlight not found: {phrase!r}')
        for r in rects:
            pg.add_highlight_annot(r).set_colors(stroke=YELLOW)
    pix = pg.get_pixmap(clip=fitz.Rect(*rect), dpi=dpi, annots=True)
    pix.save(os.path.join(OUT, name + '.png'))
    print(f'{name}: page {page + 1}, {pix.width}x{pix.height} (manual crop)')


MANUAL = [
    dict(name='ch2-christiano-fig1', arxiv_id='1706.03741', page=1, rect=(300, 318, 510, 454),
         highlight=['the reward predictor is trained asynchronously from comparisons of trajectory segments']),
    dict(name='ch2-ziegler-bug', arxiv_id='1909.08593', page=11, rect=(300, 556, 550, 714),
         highlight=['introduced a bug which flipped the sign of the reward', 'not gibberish but maximally bad output']),
    dict(name='ch2-stiennon-fig5', arxiv_id='2009.01325', page=7, rect=(100, 72, 305, 292),
         highlight=['initially improves summaries, but eventually overfits']),
    dict(name='ch2-webgpt-abstract', arxiv_id='2112.09332', page=0, rect=(136, 350, 478, 489),
         highlight=['rejection sampling against a reward model trained to predict human preferences', 'preferred by humans 56% of the time']),
    dict(name='ch2-lima-abstract', arxiv_id='2305.11206', page=0, rect=(136, 446.8, 478, 525),
         highlight=['almost all knowledge in large language models is learned during pretraining', 'only limited instruction tuning data is necessary']),
    dict(name='ch2-llama2-fig4', arxiv_id='2307.09288', page=4, rect=(70, 55, 542, 349),
         highlight=['rejection sampling and Proximal Policy Optimization']),
    dict(name='ch2-lightman-abstract', arxiv_id='2305.20050', page=0, rect=(150, 396, 462, 497),
         highlight=['process supervision significantly outperforms outcome supervision', 'PRM800K']),
    dict(name='ch2-grpo-fig4', arxiv_id='2402.03300', page=12, rect=(62, 76, 533, 306),
         highlight=['GRPO foregoes the value model']),
    dict(name='ch2-r1-aha', arxiv_id='2501.12948v1', page=8, rect=(68, 78, 528, 378),
         highlight=['Wait, wait. Wait. That’s an aha moment I can flag here', 'An interesting “aha moment” of an intermediate version of DeepSeek-R1-Zero']),
]

JOBS = [
    # ---- 2. pretrain, then prompt
    dict(name='ch2-gpt3-fig21', arxiv_id='2005.14165', page=6, figure=True, column='full',
         anchor='Figure 2.1: Zero-shot, one-shot and few-shot, contrasted with traditional', highlight=['Zero-shot, one-shot and few-shot'], below=0),
    dict(name='ch2-instructgpt-misaligned', arxiv_id='2203.02155', page=1,
         anchor='used for many recent large LMs',
         highlight=['predicting the next token on a webpage from the internet', 'follow the user’s instructions helpfully and safely',
                    'Thus, we say that the language modeling objective is misaligned'],
         above=4, end='objective is misaligned.'),
    # ---- 3. the RL-from-preferences thread
    dict(name='ch2-christiano-eq1', arxiv_id='1706.03741', page=4,
         anchor='We can interpret a reward function estimate', highlight=['This follows the Bradley-Terry model'],
         above=4, end='analogous to the famous Elo ranking system developed for chess'),
    dict(name='ch2-christiano-backflip', arxiv_id='1706.03741', page=7,
         anchor='The Hopper robot performing a sequence of backflips',
         highlight=['This behavior was trained using 900 queries in less than an hour'], above=40, end='land upright, and repeat.'),
    dict(name='ch2-ziegler-eq2', arxiv_id='1909.08593', page=2,
         anchor='Now we fine-tune', highlight=['To keep π from moving too far from ρ, we add a penalty'],
         above=4, end='encourage coherence and topicality.'),
    dict(name='ch2-stiennon-fig1', arxiv_id='2009.01325', page=1, figure=True,
         anchor='Figure 1: Fraction of the time humans prefer our models', highlight=['Human feedback'], below=0),
    dict(name='ch2-stiennon-fig2', arxiv_id='2009.01325', page=3, figure=True, column='full',
         anchor='Figure 2: Diagram of our human feedback, reward model training', highlight=['reward model training, and policy training procedure'], below=0),
    dict(name='ch2-hhh-definition', arxiv_id='2112.00861', page=2,
         anchor='We will define an AI as', highlight=['helpful, honest, and harmless'], above=20, below=20),
    # ---- 4. instruction tuning
    dict(name='ch2-flan-fig2', arxiv_id='2109.01652', page=1, figure=True, column='full',
         anchor='Figure 2: Comparing instruction tuning with pretrain', highlight=['(C) Instruction tuning (FLAN)'], below=0),
    # ---- 5. InstructGPT and its siblings
    dict(name='ch2-instructgpt-fig1', arxiv_id='2203.02155', page=1, figure=True,
         anchor='Figure 1: Human evaluations of various models on our API prompt distribution',
         highlight=['how often outputs from each model were preferred to those from the 175B SFT model'], below=0),
    dict(name='ch2-instructgpt-fig2', arxiv_id='2203.02155', page=2, figure=True, column='full',
         anchor='Figure 2: A diagram illustrating the three steps of our method', highlight=['(1) supervised fine-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)'], below=0),
    dict(name='ch2-instructgpt-eq2', arxiv_id='2203.02155', page=8,
         anchor='We also experiment with mixing the pretraining gradients', highlight=['PPO-ptx', 'control the strength of the KL penalty'],
         above=4, end='refers to the PPO-ptx models.'),
    dict(name='ch2-cai-fig1', arxiv_id='2212.08073', page=1, figure=True, column='full',
         anchor='We show the basic steps of our Constitutional AI (CAI) process',
         highlight=['supervised learning (SL) stage', 'Reinforcement Learning (RL) stage'], below=0),
    # ---- 6. 2023
    dict(name='ch2-dpo-fig1', arxiv_id='2305.18290', page=1, figure=True, column='full',
         anchor='Figure 1: DPO optimizes for human preferences while avoiding reinforcement learning',
         highlight=['DPO optimizes for human preferences while avoiding reinforcement learning'], below=0),
    dict(name='ch2-dpo-eq7', arxiv_id='2305.18290', page=3,
         anchor='Analogous to the reward modeling approach', highlight=['our policy objective becomes'],
         above=4, end='alternative parameterization, whose optimal policy is simply'),
    # ---- 7. 2024 to 2025
    dict(name='ch2-grpo-adv', arxiv_id='2402.03300', page=13,
         anchor='Outcome supervision provides the normalized reward',
         highlight=['subtracting the group average and dividing by the group standard deviation'], above=40,
         end='defined in equation (3).'),
    dict(name='ch2-tulu3-rlvr', arxiv_id='2411.15124', page=30, figure=True, column='full',
         anchor='An overview of how Reinforcement Learning with Verifiable Rewards (RLVR) works',
         highlight=['If the answer is verifiably correct, we provide reward of α, otherwise 0'], below=0),
    dict(name='ch2-r1-length', arxiv_id='2501.12948v1', page=7, figure=True, column='full',
         anchor='Figure 3 | The average response length of DeepSeek-R1-Zero', highlight=['naturally learns to solve reasoning tasks with more thinking time'], below=0),
    dict(name='ch2-r1-aime', arxiv_id='2501.12948v1', page=5,
         anchor='Performance of DeepSeek-R1-Zero Figure 2 depicts', highlight=['jumping from an initial 15.6% to an impressive 71.0%'],
         above=4, end='over time.'),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for m in MANUAL:
        if not only or m['name'] in only.split(','):
            manual(**m)
    run(JOBS)
