"""Highlighted excerpts for Chapter 6 (reinforcement learning, from scratch): Williams (1992) REINFORCE, Sutton et al. (2000)
policy gradient theorem, Ranzato et al. (2015) MIXER, Schulman et al. (2015) GAE, Ziegler et al. (2019) and Ahmadian et al. (2024).
Williams (1992) and Sutton et al. (2000) are not on arXiv. Their PDFs (the UMass copy of Williams' paper and the NeurIPS proceedings
copy of Sutton's) are expected in ~/.cache/papers as williams1992.pdf and sutton2000.pdf:
  curl -L -o ~/.cache/papers/williams1992.pdf https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf
  curl -L -o ~/.cache/papers/sutton2000.pdf https://proceedings.neurips.cc/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf
The Williams PDF has no usable text layer, so its highlights are rectangles given in points.
Usage: python shots_ch6.py   (or ONLY=name1,name2 python shots_ch6.py)"""
import os
import pymupdf as fitz
from paper_shots import run, pdf, find, YELLOW, OUT


def manual(name, arxiv_id, page, rect, highlight=(), boxes=(), dpi=190):
    """A fixed crop rectangle (points). highlight: phrases found in the text layer; boxes: rectangles (points) for scanned pages."""
    pg = pdf(arxiv_id)[page]
    for phrase in highlight:
        rects = find(pg, phrase)
        if not rects:
            raise SystemExit(f'{name}: highlight not found: {phrase!r}')
        for r in rects:
            pg.add_highlight_annot(r).set_colors(stroke=YELLOW)
    for b in boxes:
        a = pg.add_rect_annot(fitz.Rect(*b))
        a.set_colors(stroke=None, fill=YELLOW)
        a.set_opacity(0.35)
        a.set_border(width=0)
        a.update()
    pix = pg.get_pixmap(clip=fitz.Rect(*rect), dpi=dpi, annots=True)
    pix.save(os.path.join(OUT, name + '.png'))
    print(f'{name}: page {page + 1}, {pix.width}x{pix.height} (manual crop)')


MANUAL = [
    dict(name='ch6-williams-reinforce', arxiv_id='williams1992', page=4, rect=(50, 392, 565, 724),
         boxes=[(232, 488, 378, 509), (362, 575, 558, 589), (58, 589, 558, 603)]),
    dict(name='ch6-williams-episodic', arxiv_id='williams1992', page=7, rect=(50, 72, 565, 482),
         boxes=[(452, 112, 558, 126), (58, 126, 354, 140), (232, 308, 410, 342), (475, 455, 558, 469)]),
    dict(name='ch6-sutton-abstract', arxiv_id='sutton2000', page=0, rect=(125, 307, 492, 441),
         highlight=['the policy is explicitly represented by its own function approximator',
                    "Williams's REINFORCE method and actor-critic methods are examples of this approach"]),
    dict(name='ch6-sutton-variance', arxiv_id='sutton2000', page=1, rect=(105, 240, 524, 327),
         highlight=['REINFORCE learns much more slowly than RL methods using value functions',
                    'Learning a value function and using it to reduce the variance of the gradient estimate']),
]

JOBS = [
    dict(name='ch6-sutton-theorem', arxiv_id='sutton2000', page=2, column='full',
         anchor='Our first result concerns the gradient of the performance metric',
         highlight=['Theorem 1 (Policy Gradient).', 'the effect of policy changes on the distribution of states does not appear'],
         above=4, end='This is convenient for approxi'),
    dict(name='ch6-mixer-rl', arxiv_id='1511.06732', page=5, column='full',
         anchor='In order to apply the REINFORCE algorithm',
         highlight=['an action refers to predicting the next word in the sequence at each time step',
                    'Once the agent has reached the end of a sequence, it observes a reward'],
         above=4, end='expected reward. We deﬁne our loss as the negative expected reward:'),
    dict(name='ch6-gae-psi', arxiv_id='1506.02438', page=1, column='full',
         anchor='Policy gradient methods maximize the expected total reward by repeatedly',
         highlight=['TD residual', 'advantage function', 'baselined version of'],
         above=4, end='The latter formulas use the'),
    dict(name='ch6-gae-td', arxiv_id='1506.02438', page=3, column='full',
         anchor='Let V be an approximate value function.',
         highlight=['i.e., the TD residual', 'can be considered as an estimate of the', 'advantage of the action'],
         above=4, end='estimates.'),
    dict(name='ch6-gae-def', arxiv_id='1506.02438', page=4, column='full',
         anchor='The generalized advantage estimator GAE',
         highlight=['exponentially-weighted average', 'has high variance due to the sum of',
                    'introduces bias only when the value function is inaccurate'],
         above=4, end='reasonably accurate value function.'),
    dict(name='ch6-gae-fig2', arxiv_id='1506.02438', page=9, figure=True, column='full',
         anchor='Figure 2:', highlight=['The fastest policy improvement is obtain by intermediate values of'], below=0),
    dict(name='ch6-ziegler-controller', arxiv_id='1909.08593', page=2,
         anchor='Models trained with different seeds and the same KL penalty',
         highlight=['sometimes end up with quite different values of', 'log-space proportional controller'],
         above=4, end='We used Kβ = 0.1.'),
    dict(name='ch6-ziegler-table10', arxiv_id='1909.08593', page=17, figure=True, column='full',
         anchor='Table 10: Samples from a model ﬁne-tuned to mock sentiment without a KL penalty.',
         highlight=['the results are gibberish even if we include an entropy bonus'], below=30),
    dict(name='ch6-ahmadian-mdp', arxiv_id='2402.14740', page=4, column='full',
         anchor='token is modeled as an action, and partial sequences are seen as states',
         highlight=['token is modeled as an action, and partial sequences are seen as states',
                    'only generating the <EOS> token carries a reward'],
         above=14, end='spondingly shaped reward.'),
    dict(name='ch6-ahmadian-rloo', arxiv_id='2402.14740', page=5, column='full',
         anchor='REINFORCE Leave-One-Out (RLOO)',
         highlight=['The rewards for each sample can serve all other samples as a baseline',
                    'akin to a parameter-free value-function'],
         above=4, end='of fully utilizing all samples.'),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for job in MANUAL:
        if not only or job['name'] in only.split(','):
            manual(**job)
    run(JOBS)
