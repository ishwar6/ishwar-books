"""Highlighted excerpts for the Background page (before and after BERT). Each one is from the paper named in its box.
Run: python shots_partb.py   (or ONLY=pb-w2v-abstract,pb-glue-curve python shots_partb.py)"""
from paper_shots import run

JOBS = [
    # before BERT
    dict(name='pb-w2v-abstract', arxiv_id='1301.3781', page=0, anchor='We propose two novel model architectures',
         highlight=['continuous vector repre', 'sentations of words', 'less than a day', '1.6 billion words'],
         above=4, end='mance on our test set for measuring syntactic and semantic word similarities.'),
    dict(name='pb-w2v-king', arxiv_id='1301.3781', page=1, anchor='Somewhat surprisingly, it was found that similarity',
         highlight=['simple algebraic operations', 'results in a vector that is closest to the vector representation of the word Queen'],
         above=4, end='results in a vector that is closest to the vector representation of the word Queen'),
    dict(name='pb-w2v-skipgram', arxiv_id='1310.4546', page=1, anchor='The training objective of the Skip-gram model is to',
         highlight=['predicting the surrounding words', 'maximize the average', 'log probability'], above=4, below=96, snap=False),
    dict(name='pb-w2v-neg', arxiv_id='1310.4546', page=2, anchor='simplify NCE as long as the vector representations',
         highlight=['Negative sampling'], above=-2, below=56, snap=False),
    dict(name='pb-seq2seq', arxiv_id='1409.3215', page=0, anchor='Deep Neural Networks (DNNs) are powerful models',
         highlight=['multilayered Long Short-Term Memory', 'a vector of a ﬁxed dimensionality', 'decode the',
                    'target sequence from the vector'], above=4, end='target sequence from the vector.'),
    dict(name='pb-bahdanau', arxiv_id='1409.0473', page=0, anchor='Neural machine translation is a recently proposed approach',
         highlight=['ﬁxed-length vector is a', 'bottleneck', '(soft-)search', 'relevant to predicting a target word'],
         above=4, end='having to form these parts as a hard segment explicitly.'),
    dict(name='pb-bahdanau-eq5', arxiv_id='1409.0473', page=2, anchor='The context vector ci is, then, computed',
         highlight=['weighted sum of these annotations'], above=4, below=50, snap=False),
    dict(name='pb-bahdanau-eq6', arxiv_id='1409.0473', page=2, anchor='The weight αij of each annotation',
         highlight=[], above=1, below=34, snap=False),
    dict(name='pb-transformer', arxiv_id='1706.03762', page=1, anchor='Recurrent models typically factor computation',
         highlight=['inherently', 'sequential nature precludes parallelization', 'The fundamental',
                    'constraint of sequential computation, however, remains'],
         above=4, end='constraint of sequential computation, however, remains.'),
    dict(name='pb-transformer-abs', arxiv_id='1706.03762', page=0, anchor='We propose a new simple network architecture, the Transformer',
         highlight=['based solely on attention mechanisms, dispensing with recurrence and convolutions', 'more parallelizable'],
         above=4, end='less time to train.'),
    dict(name='pb-elmo', arxiv_id='1802.05365', page=0, anchor='Our representations differ from traditional word',
         highlight=['each token is assigned a', 'representation that is a function of the entire input', 'bidirec-', 'tional LSTM'],
         above=4, end='tional LSTM that is trained with a coupled lan-'),
    dict(name='pb-elmo-2', arxiv_id='1802.05365', page=0, anchor='guage model (LM) objective on a large text cor-',
         highlight=['guage model (LM) objective'], above=4,
         end='beddings from Language Models) representations.'),
    dict(name='pb-elmo-eq', arxiv_id='1802.05365', page=2, anchor='Cann et al., 2017). More generally, we compute a',
         highlight=['task speciﬁc weighting of all biLM layers'], above=-2, below=74, snap=False),
    dict(name='pb-ulmfit', arxiv_id='1801.06146', page=0, anchor='We propose Universal Language Model',
         highlight=['Universal Language Model', 'Fine-tuning (ULMFiT)', 'any task in NLP', 'ﬁne-tuning a language', 'model'],
         above=16, end='model. Our method signiﬁcantly outper-'),
    dict(name='pb-gpt1', arxiv_id='gpt1', page=0, anchor='We demonstrate that large',
         highlight=['generative pre-training of a language model', 'on a diverse corpus of unlabeled text',
                    'discriminative ﬁne-tuning'], above=4, end='speciﬁc task.'),
    dict(name='pb-gpt1-eq', arxiv_id='gpt1', page=2, anchor='Given an unsupervised corpus of tokens',
         highlight=['standard language modeling', 'objective'], above=4, below=40, snap=False),
    dict(name='pb-glue', arxiv_id='1804.07461', page=0, anchor='For natural language understanding (NLU) technology to be maximally useful',
         highlight=['must be able to process language in a way', 'not exclusive to a single task',
                    'General Language Understanding Evaluation'], above=4, end='diverse set of existing NLU tasks.'),
    # after BERT
    dict(name='pb-superglue', arxiv_id='1905.00537', page=0, anchor='In the last year, new models and methods for pretraining',
         highlight=['recently surpassed the level of non-expert', 'humans', 'limited headroom for further research'],
         above=4, end='limited headroom for further research.'),
    dict(name='pb-glue-curve', arxiv_id='1905.00537', page=1, anchor='Figure 1: GLUE benchmark performance for submitted systems',
         figure=True, column='full', below=0, highlight=['rescaled to set human performance', 'to 1.0']),
    dict(name='pb-roberta', arxiv_id='1907.11692', page=0, anchor='We present a replication study of BERT',
         highlight=['BERT', 'was signiﬁcantly undertrained', 'or exceed the performance of every model'],
         hl_nth={'BERT': 0}, above=4, end='published after it.'),
    dict(name='pb-t5', arxiv_id='1910.10683', page=0, anchor='In this paper, we explore the landscape of transfer',
         highlight=['converts all text-based', 'language problems into a text-to-text format'],
         above=16, end='language problems into a text-to-text format.'),
    dict(name='pb-gpt3-trend', arxiv_id='2005.14165', page=3, anchor='Another recent trend in language modeling may offer',
         highlight=['100 million parameters', '300 million parameters', '1.5 billion parameters', '8 billion parameters',
                    '11 billion parameters', '17 billion parameters', 'smooth trend of improvement with scale'],
         above=4, end='smooth trend of improvement with scale'),
    dict(name='pb-gpt3', arxiv_id='2005.14165', page=0, anchor='Here we show that scaling up language models',
         highlight=['175 billion', 'parameters, 10x more than any previous non-sparse language model',
                    'without any gradient updates or ﬁne-tuning'], above=4,
         end='with tasks and few-shot demonstrations speciﬁed purely via text interaction with the model.'),
    dict(name='pb-kaplan', arxiv_id='2001.08361', page=3, anchor='L(N) = (Nc/N)αN',
         highlight=['L(N) = (Nc/N)αN', 'αN ∼0.076', 'Nc ∼8.8 × 1013'], above=26, below=4, snap=False),
    dict(name='pb-deberta', arxiv_id='2006.03654', page=1, anchor='The single 1.5B-parameter DeBERTa model',
         highlight=['surpasses the human baseline (89.9 vs. 89.8) for the ﬁrst time'], above=4,
         end='decent margin (90.3 versus 89.8).'),
    dict(name='pb-modernbert', arxiv_id='2412.13663', page=0, anchor='Encoder-only transformer models such as BERT offer',
         highlight=['Encoder-only transformer models such as BERT', 'retrieval and classiﬁcation', '2 trillion tokens with a native 8192 sequence',
                    'length'], above=4, end='both single and multi-vector retrieval on dif-'),
    # after ModernBERT (2024 to 2026)
    dict(name='pb-modernbert-alt', arxiv_id='2412.13663', page=1, anchor='Alternating Attention Following recent work',
         highlight=['every third layer employs global', 'attention', '128 token, local sliding win-', 'dow attention'],
         above=4, end='dow attention with a RoPE theta of 10,000.'),
    dict(name='pb-modernbert-mlm', arxiv_id='2412.13663', page=3, anchor='use a masking rate of 30',
         highlight=['use a masking rate of 30', 'percent, as the original rate of 15 percent has since', 'been shown to be sub-optimal'],
         above=4, end='been shown to be sub-optimal (Wettig et al., 2023).'),
    dict(name='pb-neobert', arxiv_id='2502.19587', page=0, anchor='Recent innovations in architecture, pre-training',
         highlight=['encoders like BERT and RoBERTa have not seen the same level of progress', '4,096 tokens',
                    '250M parameter footprint'], above=4, end='under identical fine-tuning conditions.'),
    dict(name='pb-mmbert', arxiv_id='2509.06888', page=0, anchor='Encoder-only languages models are frequently used',
         highlight=['3T tokens of multilingual text in over 1800 languages', 'inverse mask ratio schedule'],
         above=4, end='maximizes the gains from the relatively small amount of training data.'),
    dict(name='pb-ettin', arxiv_id='2507.11412', page=0, anchor='The large language model (LLM) community focuses',
         highlight=['encoder-only models excel at classification and retrieval tasks while decoders excel at generative tasks',
                    'a 400M encoder outperforms a 1B decoder on MNLI'], above=4, end='and vice versa for generative tasks).'),
    dict(name='pb-llada', arxiv_id='2502.09992', page=0, anchor='The capabilities of large language models (LLMs) are widely',
         highlight=['forward data masking process and a reverse generation process', 'predict masked tokens',
                    'LLaDA 8B is competitive with strong LLMs like LLaMA3 8B'], above=4,
         end='instruction-following abilities in case studies such as multi-'),
    dict(name='pb-llada-eq', arxiv_id='2502.09992', page=2, anchor='The core of LLaDA is a mask predictor',
         highlight=['cross-entropy loss computed only on the masked tokens'], above=4, below=60, snap=False),
    dict(name='pb-llada-bert', arxiv_id='2502.09992', page=2, anchor='Notably, LLaDA employs a masking ratio',
         highlight=['masking ratio that varies randomly between 0 and 1 while BERT [22] uses a fixed ratio'], above=4,
         end='uses a fixed ratio.'),
    dict(name='pb-ultrabert', arxiv_id='2610.15028', page=0, anchor='Recent bidirectional encoders have been modernized',
         highlight=['Encoders saturate on downstream tasks beyond a few trillion tokens',
                    'lower masking ratios benefit classification while higher ones benefit retrieval',
                    'separate LM heads to low and high masking ratios', '91.0 on GLUE and 47.7 on BEIR'],
         above=4, end='reaching 91.0 on GLUE and 47.7 on BEIR.'),
]

# crops that need a trim after rendering: (left share, right share) of the width to keep
TRIM = {'pb-w2v-abstract': (0.22, 1), 'pb-seq2seq': (0.22, 1), 'pb-bahdanau': (0.2, 1), 'pb-glue': (0.22, 1),
        'pb-superglue': (0.22, 1), 'pb-transformer-abs': (0.22, 1), 'pb-gpt3': (0.15, 1), 'pb-bahdanau-eq5': (0, 0.68),
        'pb-neobert': (0.16, 1), 'pb-mmbert': (0.24, 1), 'pb-ettin': (0.24, 1), 'pb-llada': (0.24, 1), 'pb-ultrabert': (0.24, 1)}


def trim():
    import os
    from PIL import Image
    from paper_shots import OUT
    only = os.environ.get('ONLY')
    for name, (a, b) in TRIM.items():
        if only and name not in only.split(','):
            continue
        p = os.path.join(OUT, name + '.png')
        im = Image.open(p)
        im.crop((int(a * im.width), 0, int(b * im.width), im.height)).save(p)
    if not only or 'pb-bahdanau-eq' in only:                # Eq. 5 above Eq. 6, without the figure caption beside Eq. 5
        a, b = [Image.open(os.path.join(OUT, f'pb-bahdanau-eq{n}.png')).convert('RGB') for n in (5, 6)]
        out = Image.new('RGB', (max(a.width, b.width), a.height + b.height), 'white')
        out.paste(a, (0, 0)); out.paste(b, (0, a.height))
        out.save(os.path.join(OUT, 'pb-bahdanau-eq.png'))


if __name__ == '__main__':
    run(JOBS)
    trim()
