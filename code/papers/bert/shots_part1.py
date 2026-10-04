"""Highlighted excerpts of the BERT paper for Part 1 (title, abstract, Section 1, Figure 3)."""
from paper_shots import run

JOBS = [
    dict(name='p1-title', page=0, anchor='BERT: Pre-training of Deep Bidirectional Transformers for',
         highlight=['Bidirectional Transformers'], column='full', above=4, below=100),
    dict(name='p1-abstract-what', page=0, anchor='We introduce a new language representa',
         highlight=['Bidirectional Encoder Representations from Transformers', 'jointly conditioning on both left and right context in all layers',
                    'one additional output layer'], end='without substantial task-specific architecture modifications', above=4, below=160),
    dict(name='p1-abstract-results', page=0, anchor='BERT is conceptually simple and empirically',
         highlight=['80.5%', 'MultiNLI accuracy to 86.7%', 'Test F1 to 93.2', 'SQuAD v2.0 Test F1 to 83.1'], end='(5.1 point absolute improvement)', above=4, below=100),
    dict(name='p1-intro-tasks', page=0, anchor='Language model pre-training has been shown to',
         highlight=['Language model pre-training', 'sentence-level tasks', 'token-level tasks'], end='De Meulder, 2003; Rajpurkar et al., 2016).', above=4, below=150),
    dict(name='p1-intro-strategies', page=0, anchor='There are two existing strategies for apply',
         highlight=['feature-based and fine-tuning', 'uses task-specific architectures that include the pre-trained representations as additional features',
                    'simply fine-tuning all pre-trained parameters', 'unidirectional language models'], end='to learn general language representations.', above=4, below=170),
    dict(name='p1-intro-limitation', page=0, anchor='We argue that current techniques restrict the',
         highlight=['standard language models are unidirectional', 'every token can only attend to previous tokens',
                    'crucial to incorporate context from both directions'], end='context from both directions.', above=4, below=180),
    dict(name='p1-intro-mlm', page=0, anchor='In this paper, we improve the fine-tuning based',
         highlight=['masked language model', 'randomly masks some of the tokens from the input', 'predict the original vocabulary id of the masked'],
         end='predict the original vocabulary id of the masked', above=4, below=200),
    dict(name='p1-intro-mlm2', page=1, anchor='word based only on its context',
         highlight=['word based only on its context', 'fuse the left and the right context', 'next sentence prediction'], end='The contributions of our paper are as follows:', above=4, below=60),
    dict(name='p1-contributions', page=1, anchor='We demonstrate the importance of bidirectional',
         highlight=['We demonstrate the importance of bidirectional pre-training', 'reduce the need for many heavily-engineered task-specific architectures',
                    'BERT advances the state of the art for eleven NLP tasks'], end='google-research/bert.', above=4, below=230),
    dict(name='p1-figure3', page=12, anchor='Figure 3: Differences in pre-training model architectures',
         highlight=['Among the three, only BERT representations are jointly conditioned on both left and right context in all layers'],
         figure=True, column='full', below=0),
    dict(name='p1-transformer-fig1', arxiv_id='1706.03762', page=2, anchor='Figure 1: The Transformer - model architecture.',
         highlight=['Figure 1: The Transformer - model architecture.'], figure=True, column='full', below=0),
    dict(name='p1-transformer-halves', arxiv_id='1706.03762', page=2, anchor='The Transformer follows this overall architecture',
         highlight=['encoder and decoder, shown in the left and right halves of Figure 1', 'prevent positions from attending to subsequent positions',
                    'can depend only on the known outputs at positions less than'], end='at positions less than i.', above=4, below=200),
    dict(name='p1-elmo-bilm', arxiv_id='1802.05365', page=2, anchor='Recent state-of-the-art neural language models',
         highlight=['predicting the previous token given the future context', 'jointly maximizes the log likelihood of the forward and backward directions',
                    'separate parameters for the LSTMs in each direction'], end='for the LSTMs in each direction', above=80, below=200),
]

if __name__ == '__main__':
    run(JOBS)
