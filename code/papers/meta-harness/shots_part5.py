"""Highlighted excerpts for Part 5 (Limits, and What Came After): §5, §4.3, §3, footnote 1, Appendix A.2, Appendix D,
Figure 7, and the later papers on harness search (each from its own arXiv PDF).
Run: python shots_part5.py   (ONLY=name1,name2 to redo some)   ->  public/img/papers/meta-harness/p5-*.png"""
from paper_shots import run

AHE, SELF, RHI, RETHINK, HZERO = '2604.25850', '2606.09498', '2607.15524', '2607.12227', '2609.24974'

JOBS = [
    # ---- Meta-Harness, §5 Discussion (page 10 = index 9)
    dict(name='p5-discussion', page=9, anchor='Beyond outperforming existing harnesses',
         highlight=['generalize to out-of-distribution classification datasets', 'unseen base models in the math setting',
                    'a few hours', 'readable, transferable strategies'],
         end='including future, stronger ones.'),
    dict(name='p5-overfit-code', page=9, anchor='Overfitting in code space is also more',
         highlight=['Overfitting in code space is also more', 'inspectable', 'brittle if-chains or hard-coded class mappings',
                    'weight-space overfitting is not'],
         end='weight-space overfitting is not.', above=12),
    dict(name='p5-figure7', page=19, anchor='Figure 7: Search-set vs. test accuracy', figure=True,
         highlight=['Search-set vs. test accuracy per dataset', 'The dashed diagonal is y=x'], below=0),
    # ---- §4.3: search and final evaluation on the same 89 tasks
    dict(name='p5-tbench-same', page=8, anchor='Terminus 2 [33] and Terminus-KIRA [25]. For this',
         highlight=['we perform search and final evaluation on the same 89-task benchmark', 'discovery problem',
                    'regex-based audits for task-specific string leakage'],
         end='task-specific string leakage into evolved harnesses.', above=4),
    # ---- cost: §3 practical implementation, §1 tokens per evaluation
    dict(name='p5-cost-run', page=4, anchor='a typical run evaluates roughly 60 harnesses over 20 iterations',
         highlight=['a typical run evaluates roughly 60 harnesses over 20 iterations'],
         above=40, below=4),
    dict(name='p5-cost-tokens', page=2, anchor='study, a single evaluation can produce up to 10,000,000 tokens',
         highlight=['a single evaluation can produce up to 10,000,000 tokens of diagnostic information'],
         end='optimization settings (Table 1).', above=4),
    # ---- Appendix D: search-set size and the skill
    dict(name='p5-appd-skill', page=23, anchor='Write a good skill.',
         highlight=['iterating on the skill text had a larger effect on search quality than changing iteration count or population size',
                    'a few short evolution runs (3–5 iterations each)'],
         end='before committing to a full run.', above=4),
    dict(name='p5-appd-set-a', page=23, anchor='Start with a baseline harness and a search set',
         highlight=['filtering for examples', 'that the baseline gets wrong', 'The search has little to optimize if the baseline already saturates the evaluation',
                    'roughly 50 full evaluations per run (50–100 examples in our'],
         end='50 full evaluations per run', above=4, below=4),
    dict(name='p5-appd-set-b', page=24, anchor='classification experiments, 88 problems for math retrieval',
         highlight=['88 problems for math retrieval', 'a fast, discriminative eval is', 'more valuable than a large one'],
         above=4, below=14),
    # ---- one proposer: §5 end and footnote 1
    dict(name='p5-one-proposer', page=9, anchor='weights, letting the strategy shape what the model learns',
         highlight=['one particularly strong coding-agent proposer', 'how the effect varies across proposer agents remains for future work'],
         end='remains for future work.', above=2),
    dict(name='p5-footnote', page=3, anchor='earlier exploration, we think this workflow only became practical',
         highlight=['only became practical recently', 'around early 2026'], above=3, below=10),
    # ---- Appendix A.2: the TerminalBench-2 search log
    dict(name='p5-a2-start', page=15, anchor='Iterations 1–2: promising bugfixes are confounded by prompt edits',
         highlight=['both regress', 'sharply from the 64.4% Terminus-KIRA baseline'],
         end='sharply from the 64.4% Terminus-KIRA baseline.', above=4),
    dict(name='p5-a2-iter3', page=15, anchor='Iteration 3: the proposer identifies the confound',
         highlight=['Root cause of regressions', 'The structural bugfixes were confounded with harmful prompt changes',
                    '(63.3%, -1.1pp)'],
         end='which supports the confound diagnosis.', above=4),
    dict(name='p5-a2-iter7', page=16, anchor='Iteration 7: the winning candidate',
         highlight=['purely additive', 'without risking regression on already-passing tasks', 'This candidate is the best result so far'],
         above=4, below=158),
    dict(name='p5-a2-iter10', page=16, anchor='Iteration 10: cross-run transfer',
         highlight=['worth +18pp', 'crashed before evaluation'], end='crashed before evaluation.', above=4),
    # ---- §5: co-evolve harness and weights
    dict(name='p5-coevolve', page=9, anchor='Our findings reflect a recurring pattern in machine learning',
         highlight=['stronger general-purpose agents can outperform hand-engineered', 'co-evolve the harness and the model weights'],
         end='letting the strategy shape what the model learns and vice versa.', above=4),

    # ---- later work, each from its own paper
    dict(name='p5-ahe-abstract', arxiv_id=AHE, page=0, anchor='effect is hard to attribute. We introduce',
         highlight=['component observability', 'experience', 'observability distills millions of raw trajectory tokens',
                    'decision', 'observability pairs every edit with a self-declared prediction', 'from 69.7% to 77.0%'],
         end='2 from 69.7% to 77.0%', above=4,
         hl_nth={'experience': 0, 'decision': 0}),
    dict(name='p5-ahe-blind', arxiv_id=AHE, page=8, anchor='Regression blindness.',
         highlight=['regression-', 'precision of 11.8% and regression-recall of 11.1%', 'most upcoming regressions go unforeseen'],
         end='produces the non-monotone steps in the evolution curve', above=4, hl_nth={'regression-': 0}),
    dict(name='p5-self-abstract', arxiv_id=SELF, page=0, anchor='In this paper, we introduce Self-Harness',
         highlight=['an LLM-based agent', 'improves its own operating harness, without relying on human engineers or stronger external agents',
                    'Weakness Mining', 'Harness Proposal', 'Proposal Validation'],
         end='edits only after regression testing.', above=4),
    dict(name='p5-self-rule', arxiv_id=SELF, page=7, anchor='A candidate is accepted only if it improves at least one split',
         highlight=['A candidate is accepted only if it improves at least one split without degrading the other',
                    'Proposals that only trade off one split against', 'the other are rejected'],
         end='the other are rejected, even if their total pass count increases.', above=4),
    dict(name='p5-rhi-abstract', arxiv_id=RHI, page=0, anchor='To this end, we introduce Recursive',
         highlight=['represents the harness as a prompt-level specification of the agent loop',
                    'pairwise feedback over its own revision history', 'reducing inference', 'cost by up to 60%'],
         end='rather than longer reasoning', above=4, hl_nth={'reducing inference': 0}),
    dict(name='p5-rhi-mh-a', arxiv_id=RHI, page=5, anchor='This distinction separates Meta-Harness',
         highlight=['Meta-Harness is better viewed as', 'optimizing the finite-candidate approximation in Equation (2)'],
         end='scores, and execution', above=4),
    dict(name='p5-rhi-mh', arxiv_id=RHI, page=6, anchor='traces in a filesystem, and imposes no parent-selection rule',
         highlight=['It therefore does not', 'instantiate Equation (3), whose comparator is only the immediately previous harness'],
         end='instantiate Equation (3), whose comparator', above=4, below=4),
    dict(name='p5-rhi-cost', arxiv_id=RHI, page=6, anchor='Table 1 | Per-iteration computational cost',
         highlight=['Per-iteration computational cost of three harness-search objectives'],
         end='Trajectory-local RHI (Eq. 3)', above=4, below=4),
    dict(name='p5-rethink-abstract', arxiv_id=RETHINK, page=0, anchor='Second, prior work searches for harness configurations',
         highlight=['violating the standard separation', 'between training and test data',
                    'fails to outperform simple test-time scal-', 'exhibits limited generalization'],
         end='exhibits limited generalization.', above=4),
    dict(name='p5-rethink-table3', arxiv_id=RETHINK, page=6, anchor='Table 3: Experimental results on Terminal-Bench 2.1 with disjoint',
         highlight=['disjoint search and evaluation tasks', 'Harness Evolution yields no significant gains'],
         end='a strong tendency to overfit the search set.', above=70),
    dict(name='p5-hzero-abstract', arxiv_id=HZERO, page=0, anchor='We therefore study agent harness distillation',
         highlight=['agent harness distillation', 'transferring the behaviors it induces', 'into model weights',
                    'macro-average task success from 23.3% to 44.3%', 'even exceeding the 41.7% it reaches with that harness still',
                    'attached'],
         end='attached.', above=4, hl_nth={'attached': 0}),
]

# arXiv's rotated side stamp sits in the left margin of some first pages: cut it off (pixels from the left)
TRIM_LEFT = {'p5-ahe-abstract': 300, 'p5-rhi-abstract': 115, 'p5-rethink-abstract': 300, 'p5-hzero-abstract': 140,
             'p5-self-abstract': 120}


def trim():
    import os
    from PIL import Image
    from paper_shots import OUT
    only = os.environ.get('ONLY')
    for name, px in TRIM_LEFT.items():
        if only and name not in only.split(','):
            continue
        path = os.path.join(OUT, name + '.png')
        im = Image.open(path)
        im.crop((px, 0, im.width, im.height)).save(path)
        print(f'{name}: trimmed {px} px on the left')


if __name__ == '__main__':
    run(JOBS)
    trim()
