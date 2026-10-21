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
    # Part 3: reinforcement learning
    dict(name='h-sutton88', arxiv_id='sutton1988', page=0, anchor='This article introduces a class of incremental learning procedures',
         highlight=['difference between temporally', 'successive predictions', "Samuel's checker player"], above=4,
         end='difference methods can be applied to advantage.'),
    dict(name='h-dqn-fig', arxiv_id='dqn_nature', page=1, anchor='Figure 1 | Schematic illustration of the convolutional neural network.',
         figure=True, column='full', below=0, highlight=['Schematic illustration of the convolutional neural network.']),
    dict(name='h-alphago-fig', arxiv_id='alphago2016', page=1, anchor='Figure 1 | Neural network training pipeline and architecture.',
         figure=True, column='full', below=0, highlight=['Neural network training pipeline and architecture.']),
    dict(name='h-agz-abs', arxiv_id='agz2017', page=0, anchor='A long-standing goal of artificial intelligence is an algorithm',
         highlight=['tabula rasa', 'without hu-', 'man data, guidance, or domain knowledge beyond game rules', 'AlphaGo becomes its own',
                    '100-0 against the previously published'], above=4,
         end='champion-defeating AlphaGo.'),
    dict(name='h-ppo-clip', arxiv_id='1707.06347', page=2, anchor='The main objective we propose is the following',
         highlight=['removes the incentive for moving', 'pessimistic bound'], above=4,
         end='when it makes the objective worse.'),
    dict(name='h-ppo-fig', arxiv_id='1707.06347', page=2, anchor='Figure 1: Plots showing one term',
         figure=True, column='full', below=0, highlight=['for positive advantages (left) and negative advantages (right)']),
    dict(name='h-alphazero', arxiv_id='1712.01815', page=0, anchor='The game of chess is the most widely-studied domain',
         highlight=['given no domain knowledge except the game rules', 'within 24 hours'], above=4,
         end='defeated a world-champion program in each case.'),
    dict(name='h-dactyl', arxiv_id='1808.00177', page=0, anchor='We use reinforcement learning (RL) to learn dexterous',
         highlight=['randomize many of the physical properties', 'trained entirely in simulation'], above=4,
         end='the controlled use of gravity.'),
    # Part 4: how language models are trained today
    dict(name='h-christiano', arxiv_id='1706.03741', page=0, anchor='For sophisticated reinforcement learning (RL) systems',
         highlight=['human preferences between pairs of trajectory segments', 'less than 1% of our agent’s interactions',
                    'about an hour of human time'], above=4, end='previously learned from human feedback.'),
    dict(name='h-stiennon-fig', arxiv_id='2009.01325', page=1, anchor='Figure 1: Fraction of the time humans prefer our models',
         figure=True, below=0, highlight=['Fraction of the time humans prefer our models’ summaries over the human-generated reference summaries']),
    dict(name='h-gao-fig', arxiv_id='2210.10760', page=2, anchor='Figure 1: Reward model (RM) parameter size scaling experiments',
         figure=True, column='full', below=0, fig_top=262, highlight=['the gold reward initially increases and later decreases']),
    dict(name='h-llama2-fig', arxiv_id='2307.09288', page=4, anchor='Figure 4: Training of Llama 2-Chat',
         figure=True, column='full', below=0, fig_top=226, highlight=['rejection sampling and Proximal Policy Optimization (PPO)']),
    dict(name='h-dpo-fig', arxiv_id='2305.18290', page=1, anchor='Figure 1: DPO optimizes for human preferences while avoiding reinforcement learning.',
         figure=True, column='full', below=0, highlight=['DPO optimizes for human preferences while avoiding reinforcement learning.']),
    dict(name='h-lightman', arxiv_id='2305.20050', page=0, anchor='In recent years, large language models have greatly improved',
         highlight=['process supervision significantly outper-', 'forms outcome supervision', '78% of problems'],
         above=4, end='used to train our best reward model.'),
    dict(name='h-grpo-fig', arxiv_id='2402.03300', page=12, anchor='Figure 4 | Demonstration of PPO and our GRPO.',
         figure=True, column='full', below=22, highlight=['GRPO foregoes the value model, instead estimating the baseline from group scores']),
    dict(name='h-r1-fig', arxiv_id='2501.12948', page=3, anchor='Figure 1 | (a) AIME accuracy of DeepSeek-R1-Zero during training.',
         figure=True, column='full', below=86, highlight=['DeepSeek-R1-Zero naturally learns to solve reasoning tasks with more thinking time.']),
    dict(name='h-tulu3-rlvr', arxiv_id='2411.15124', page=30, anchor='Figure 18 An overview of how Reinforcement Learning with Verifiable Rewards',
         figure=True, column='full', below=0, highlight=['Reinforcement Learning with Verifiable Rewards (RLVR)']),
    dict(name='h-gemini-post', arxiv_id='2507.06261', page=4, anchor='Since the initial announcement of Gemini 1.5, significant advancements',
         highlight=['Fine-Tuning (SFT), Reward Modeling (RM), and Reinforcement Learning (RL) stages',
                    'increased the training compute allocated to RL', 'verifiable rewards',
                    'model-based generative rewards', 'multi-step actions and tool use',
                    'Gemini Thinking models are trained with Reinforcement Learning to use additional compute at',
                    'inference time to arrive at more accurate answers.'],
         above=18, end='inference time to arrive at more accurate answers.'),
    dict(name='h-olmo3-flow', arxiv_id='2512.13961', page=3, anchor='Figure 2 Depiction of model flow for Olmo 3.',
         figure=True, column='full', below=0, highlight=['Depiction of model flow for Olmo 3.']),
    dict(name='h-qwen3-flow', arxiv_id='2505.09388', page=8, anchor='Figure 1: Post-training pipeline of the Qwen3 series models',
         figure=True, column='full', below=0, highlight=['Post-training pipeline of the Qwen3 series models']),
    dict(name='h-dsv32', arxiv_id='2512.02556', page=1, anchor='Second, we develop a stable and scalable RL protocol',
         highlight=['post-training computational budget exceeding 10% of the pre-training cost'], above=4,
         end='unlocking advanced capabilities.'),
    dict(name='h-delib', arxiv_id='2412.16339', page=0, anchor='As large-scale language models increasingly impact',
         highlight=['directly teaches the model safety specifications', 'explicitly recall and accurately reason over the specifications before answering'],
         above=4, end='more scalable, trustworthy, and interpretable alignment.'),
    # Part 5: inside a modern language model
    dict(name='h-kaplan-fig', arxiv_id='2001.08361', page=2, anchor='Language modeling performance improves smoothly as we increase the model size',
         figure=True, column='full', below=0, highlight=['For optimal performance all three factors must be scaled']),
    dict(name='h-chinchilla-abs', arxiv_id='2203.15556', page=0, anchor='We investigate the optimal model size and number of tokens',
         highlight=['current large language models are signiﬁcantly under-', 'the model size and',
                    'the number of training tokens should be scaled equally', 'with 70B parameters and', '4× more more data'],
         above=4, end='Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks.'),
    dict(name='h-llama3-scale', arxiv_id='2407.21783', page=0, anchor='Scale. We train a model at far larger scale than previous Llama models',
         highlight=['3.8 × 1025 FLOPs', '405B trainable parameters on 15.6T text tokens'],
         above=4, end='15.6T text tokens. As expected per'),
    dict(name='h-llama3-over', arxiv_id='2407.21783', page=1, anchor='scaling laws for foundation models, our flagship model outperforms smaller models',
         highlight=['we also train our smaller models for much longer than is compute-optimal',
                    'perform better than compute-optimal models at the same inference budget'],
         above=4, end='quality of those smaller models during post-training.'),
    dict(name='h-fineweb-abs', arxiv_id='2406.17557', page=0, anchor='The performance of a large language model (LLM) depends heavily on the quality',
         highlight=['15-trillion token dataset derived from 96 Common Crawl snapshots', '1.3-trillion token collection of educational text'],
         above=4, end='all of the models trained during our ablation experiments.'),
    dict(name='h-fineweb-dedup', arxiv_id='2406.17557', page=4, anchor='data). Results are presented in Fig. 4.',
         highlight=['was actually of worse quality than the 90% of data', 'individually deduplicating each snapshot'],
         above=4, end='matched RefinedWeb’s performance, as per Fig. 5.'),
    dict(name='h-rope-fig', arxiv_id='2104.09864', page=4, anchor='Figure 1: Implementation of Rotary Position Embedding(RoPE).',
         figure=True, column='full', below=0, highlight=['Implementation of Rotary Position Embedding(RoPE).']),
    dict(name='h-mla-fig', arxiv_id='2405.04434', page=6, anchor='Figure 3 | Simplified illustration of Multi-Head Attention',
         figure=True, column='full', below=0, highlight=['jointly compressing the keys and values into a latent vector']),
    dict(name='h-dsv3-cost-a', arxiv_id='2412.19437', page=4, anchor='Table 1 | Training costs of DeepSeek-V3',
         highlight=['2788K', '$5.576M'], above=64, end='assuming the rental price of H800 is $2 per GPU hour.'),
    dict(name='h-dsv3-cost-b', arxiv_id='2412.19437', page=4, anchor='Lastly, we emphasize again the economical training costs',
         highlight=['180K', 'H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs', 'excluding the costs',
                    'associated with prior research and ablation experiments'],
         above=4, end='associated with prior research and ablation experiments on architectures, algorithms, or data.'),
    dict(name='h-mixtral-fig', arxiv_id='2401.04088', page=1, anchor='Figure 1: Mixture of Experts Layer.',
         figure=True, column='full', below=0, highlight=['Each input vector is assigned to 2 of the 8 experts by a router.']),
    dict(name='h-swiglu', arxiv_id='2002.05202', page=2, anchor='We have extended the GLU family of layers',
         highlight=['We oﬀer no explanation as to why these', 'architectures seem to work; we attribute their success, as all else, to divine benevolence.'],
         above=4, end='to divine benevolence.'),
    dict(name='h-vit-fig', arxiv_id='2010.11929', page=2, anchor='Figure 1: Model overview.',
         figure=True, column='full', below=0, highlight=['We split an image into ﬁxed-size patches']),
    dict(name='h-gemini25-tpu', arxiv_id='2507.06261', page=3, anchor='This model family is the first to be trained on TPUv5p architecture.',
         highlight=['multiple 8960-chip pods of Google’s TPUv5p accelerators', 'interruptions from hardware failures multiple times per hour'],
         above=18, end='machinery is designed to tolerate the higher failure rates expected at much larger scales.'),
    dict(name='h-gemini25-distill', arxiv_id='2507.06261', page=2, anchor='The smaller models in the Gemini 2.5 series',
         highlight=['Flash size and below', 'use distillation', 'k-sparse distribution over the vocabulary'],
         above=4, end='reduced serving cost (see Figure 2).'),
]

# PDFs with a broken text layer: (name, pdf id, 0-based page, crop rect in points, highlight rects in points)
MANUAL = [
    ('h-dartmouth', 'dartmouth1955', 1, (66, 62, 548, 178), [(224, 100, 548, 112), (66, 112, 446, 124)]),
    ('h-clip', '2103.00020', 1, (50, 64, 546, 298), ['CLIP jointly trains an image encoder and a text encoder to predict the correct pairings']),
    ('h-lenet', 'lenet1998', 6, (36, 48, 580, 248), []),
    ('h-lstm', 'lstm1997', 6, (64, 420, 532, 652), []),
    # DeepSeek-R1 Table 2: the 'aha moment' (maths symbols break the automatic cut, so the crop is by coordinates)
    ('h-r1-aha', '2501.12948', 4, (70, 82, 527, 368), ['Wait, wait. Wait.', 'aha moment']),
    # MuZero Figure 1 with only the first part of its long caption (panel A), so the crop does not run off the page
    # Part 5: figures whose automatic crop caught a running header or cut the caption, cut by page coordinates
    ('h-gemini25-arch', '2507.06261', 1, (52, 559, 545, 752), ['The Gemini 2.5 models are sparse mixture-of-experts (MoE)',
        'with native multimodal support for text,', 'vision, and audio inputs', 'decouple total model capacity from computation and serving cost per token',
        'training instabilities']),
    ('h-megatron-fig', '1909.08053', 3, (297, 58, 551, 348), ['Blocks of Transformer with Model Parallelism.']),
    ('h-dsv3-fig', '2412.19437', 6, (61, 76, 536, 484), ['Illustration of the basic architecture of DeepSeek-V3.']),
    ('h-flash-fig', '2205.14135', 1, (61, 66, 552, 322), ['uses tiling to prevent materialization of the large']),
    ('h-switch-fig', '2101.03961', 4, (80, 85, 532, 399), ['We replace the dense feed', 'sparse Switch FFN layer']),
    ('h-bpe-alg', '1508.07909', 3, (62, 57.6, 300, 305.4), ['best = max(pairs, key=pairs.get)']),
    ('h-muzero-fig', '1911.08265', 2, (67.67, 74.52, 541.22, 370.5), ['Planning, acting, and training with a learned model.']),
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


# Part 5: two cuts from one page (or two pages) shown as one excerpt
STITCH = {'h-llama3-scale': ['h-llama3-scale', 'h-llama3-over'], 'h-dsv3-cost': ['h-dsv3-cost-a', 'h-dsv3-cost-b']}


def stitch():
    from PIL import Image
    only = os.environ.get('ONLY')
    for out, parts in STITCH.items():
        if only and not any(x in only.split(',') for x in parts + [out]):
            continue
        ims = [Image.open(os.path.join(paper_shots.OUT, n + '.png')).convert('RGB') for n in parts]
        W = max(i.width for i in ims)
        canvas = Image.new('RGB', (W, sum(i.height for i in ims) + 24 * (len(ims) - 1)), 'white')
        y = 0
        for i in ims:
            canvas.paste(i, (0, y)); y += i.height + 24
        canvas.save(os.path.join(paper_shots.OUT, out + '.png'))
        for n in parts:
            if n != out:
                os.remove(os.path.join(paper_shots.OUT, n + '.png'))


TRIM = {'h-gpt3': (0.15, 1), 'h-bahdanau': (0.2, 1), 'h-instructgpt-abs': (0.18, 1),
        'h-react': (0.18, 1), 'h-cai': (0.12, 1), 'h-cot': (0.1, 1), 'h-tabular': (0.18, 1),
        'h-alphazero': (0.15, 1), 'h-dactyl': (0.22, 1), 'h-christiano': (0.22, 1), 'h-lightman': (0.27, 1), 'h-delib': (0.13, 1),
        'h-chinchilla-abs': (0.07, 1), 'h-fineweb-abs': (0.2, 1), 'h-llama3-scale': (0.08, 1)}


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
    stitch()
