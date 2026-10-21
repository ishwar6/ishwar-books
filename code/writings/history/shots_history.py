"""Highlighted paper excerpts for the "How AI Got Here" series (writings/from-neuron-to-opus, writings/classical-machine-learning).
Uses the shared cutter in code/papers/bert/paper_shots.py; writes PNGs to public/img/writings/history/.
Non-arXiv PDFs live in ~/.cache/papers/<id>.pdf (downloaded from the authors' or publishers' own pages).
Old PDFs whose text layer is broken (LeNet 1998, LSTM 1997, the Dartmouth proposal), and figures the automatic cut misses (CLIP),
are cut by page coordinates instead.
Run: python shots_history.py   (or ONLY=h-resnet,h-clip python shots_history.py)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers', 'bert'))
import paper_shots
from paper_shots import shot, pdf, YELLOW
paper_shots.OUT = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'writings', 'history')
import pymupdf as fitz

JOBS = [
    # Part 1: the neural line
    dict(name='h-alexnet-fig', arxiv_id='alexnet2012', page=4, anchor='Figure 2: An illustration of the architecture of our CNN',
         figure=True, column='full', below=0, highlight=['between the two GPUs']),
    dict(name='h-bahdanau', arxiv_id='1409.0473', page=0, anchor='Neural machine translation is a recently proposed approach',
         highlight=['ﬁxed-length vector is a', 'bottleneck', '(soft-)search', 'relevant to predicting a target word'],
         above=4, end='having to form these parts as a hard segment explicitly.'),
    dict(name='h-resnet', arxiv_id='1512.03385', page=1, anchor='Figure 2. Residual learning: a building block.',
         figure=True, below=0, highlight=['Residual learning: a building block.']),
    dict(name='h-transformer', arxiv_id='1706.03762', page=2, anchor='Figure 1: The Transformer - model architecture.',
         figure=True, column='full', below=0, highlight=['The Transformer - model architecture.']),
    dict(name='h-gpt3', arxiv_id='2005.14165', page=0, anchor='Here we show that scaling up language models',
         highlight=['175 billion', 'parameters, 10x more than any previous non-sparse language model',
                    'without any gradient updates or ﬁne-tuning'], above=4,
         end='with tasks and few-shot demonstrations speciﬁed purely via text interaction with the model.'),
    dict(name='h-instructgpt-abs', arxiv_id='2203.02155', page=0, anchor='Making language models bigger does not inherently',
         highlight=['does not inherently make them better at following a user’s intent',
                    'outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3'],
         above=4, end='despite having 100x fewer parameters.'),
    dict(name='h-instructgpt-fig', arxiv_id='2203.02155', page=2, anchor='Figure 2: A diagram illustrating the three steps of our method',
         figure=True, column='full', below=0, highlight=['(1) supervised ﬁne-tuning (SFT), (2) reward model (RM) training, and (3) reinforcement learning via proximal policy optimization (PPO)']),
    dict(name='h-cot', arxiv_id='2201.11903', page=0, anchor='Figure 1: Chain-of-thought prompting enables',
         figure=True, column='full', below=0, highlight=['Chain-of-thought reasoning processes are highlighted.']),
    dict(name='h-react', arxiv_id='2210.03629', page=0, anchor='While large language models (LLMs) have demonstrated',
         highlight=['generate both reasoning traces and task-speciﬁc actions in an interleaved manner'],
         above=4, end='such as knowledge bases or environments.'),
    dict(name='h-cai', arxiv_id='2212.08073', page=0, anchor='As AI systems become more capable, we would like to enlist',
         highlight=['The only human oversight is provided through a list of rules or principles', '‘Constitutional AI’',
                    '‘RL from AI Feedback’ (RLAIF)'], above=4, end='‘RL from AI Feedback’ (RLAIF).'),
    # Part 2: classical machine learning
    dict(name='h-rf', arxiv_id='breiman2001', page=0, anchor='Random forests are a combination of tree predictors',
         highlight=['combination of tree predictors', 'random selection of features to split each node',
                    'compare favorably to Adaboost'], above=4, end='but are more robust with respect to noise.'),
    dict(name='h-xgb-abs', arxiv_id='1603.02754', page=0, anchor='Tree boosting is a highly eﬀective and widely used',
         highlight=['scalable end-', 'to-end tree boosting system called XGBoost', 'scales beyond billions of examples'],
         above=4, end='using far fewer resources than existing systems.'),
    dict(name='h-xgb-kaggle', arxiv_id='1603.02754', page=0, anchor='Among the 29 challenge winning solutions',
         highlight=['Among the 29 challenge winning solutions', '17 solutions used XGBoost'], above=4,
         end='17 solutions used XGBoost.'),
    dict(name='h-tabular', arxiv_id='2207.08815', page=0, anchor='While deep learning has enabled tremendous progress',
         highlight=['45 datasets', 'tree-based models remain state-of-the-art on medium-sized data'],
         above=4, end='even without accounting for their superior speed.'),
]

# PDFs with a broken text layer: (name, pdf id, 0-based page, crop rect in points, highlight rects in points)
MANUAL = [
    ('h-dartmouth', 'dartmouth1955', 1, (66, 62, 548, 178), [(224, 100, 548, 112), (66, 112, 446, 124)]),
    ('h-clip', '2103.00020', 1, (50, 64, 546, 298), ['CLIP jointly trains an image encoder and a text encoder to predict the correct pairings']),
    ('h-lenet', 'lenet1998', 6, (36, 48, 580, 248), []),
    ('h-lstm', 'lstm1997', 6, (64, 420, 532, 652), []),
]


def manual(name, pid, page, crop, rects, dpi=190):
    pg = pdf(pid)[page]
    for r in rects:
        if isinstance(r, str):                         # a phrase: the text layer works, so highlight it properly
            for h in paper_shots.find(pg, r):
                pg.add_highlight_annot(h).set_colors(stroke=YELLOW)
        else:
            pg.draw_rect(fitz.Rect(r), color=None, fill=YELLOW, fill_opacity=0.38, overlay=True)
    pix = pg.get_pixmap(clip=fitz.Rect(crop), dpi=dpi, annots=True)
    os.makedirs(paper_shots.OUT, exist_ok=True)
    pix.save(os.path.join(paper_shots.OUT, name + '.png'))
    print(f'{name}: page {page + 1}, {pix.width}x{pix.height}')


TRIM = {'h-gpt3': (0.15, 1), 'h-bahdanau': (0.2, 1), 'h-instructgpt-abs': (0.18, 1),
        'h-react': (0.18, 1), 'h-cai': (0.12, 1), 'h-cot': (0.1, 1), 'h-tabular': (0.18, 1)}


def trim():
    from PIL import Image
    only = os.environ.get('ONLY')
    for name, (a, b) in TRIM.items():
        if only and name not in only.split(','):
            continue
        p = os.path.join(paper_shots.OUT, name + '.png')
        im = Image.open(p)
        im.crop((int(a * im.width), 0, int(b * im.width), im.height)).save(p)


if __name__ == '__main__':
    only = os.environ.get('ONLY')
    paper_shots.run(JOBS)
    for m in MANUAL:
        if not only or m[0] in only.split(','):
            manual(*m)
    trim()
