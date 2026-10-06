import os
"""Highlighted excerpts of the ViT paper for Part 1 (title, authors, abstract, Section 1, the code footnote),
plus the first sentence of the Transformer abstract and the first paragraph of the BERT abstract."""
from paper_shots import run

JOBS = [
    dict(name='p1-title', page=0, anchor='AN IMAGE IS WORTH 16X16 WORDS:',
         highlight=['16X16 WORDS', 'IMAGE RECOGNITION AT SCALE'], column='full', above=4, end='TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE'),
    dict(name='p1-authors', page=0, anchor='Alexey Dosovitskiy',
         highlight=['equal technical contribution', 'equal advising', 'Google Research, Brain Team'], column='full', above=4,
         end='{adosovitskiy, neilhoulsby}@google.com'),
    dict(name='p1-abstract-1', page=0, anchor='While the Transformer architecture has become the de-facto standard',
         highlight=['de-facto standard for natural language processing tasks', 'applications to computer vision remain limited',
                    'in conjunction with convolutional networks', 'replace certain components of convolutional networks'],
         column='full', above=4, end='overall structure in place.'),
    dict(name='p1-abstract-2', page=0, anchor='We show that this reliance on CNNs is not necessary',
         highlight=['reliance on CNNs is not necessary', 'pure transformer applied directly to sequences of image patches',
                    'pre-trained on large amounts of data', 'ImageNet, CIFAR-100, VTAB', 'fewer computational resources to train.1'],
         column='full', above=4, end='resources to train.1'),
    dict(name='p1-intro-nlp', page=0, anchor='Self-attention-based architectures, in particular Transformers',
         highlight=['model of choice in natural language processing', 'pre-train on a large text corpus and then fine-tune on a smaller task-specific dataset',
                    'over 100B parameters', 'no sign of saturating performance'],
         column='full', above=4, end='no sign of saturating performance.'),
    dict(name='p1-intro-cnn', page=0, anchor='In computer vision, however, convolutional architectures remain dominant',
         highlight=['convolutional architectures remain dominant', 'combining CNN-like architectures with self-attention', 'replacing the convolutions entirely',
                    'specialized attention patterns', 'classic ResNet-like architectures are still state of the art'],
         column='full', above=4, end='Kolesnikov et al., 2020).'),
    dict(name='p1-intro-ours', page=0, anchor='Inspired by the Transformer scaling successes in NLP',
         highlight=['standard Transformer directly to images, with the fewest possible modifications', 'split an image into patches',
                    'sequence of linear embeddings of these patches', 'treated the same way as tokens (words)', 'image classification in supervised fashion'],
         column='full', above=4, end='in supervised fashion.'),
    dict(name='p1-intro-midsize', page=0, anchor='When trained on mid-sized datasets such as ImageNet',
         highlight=['mid-sized datasets such as ImageNet without strong regularization', 'a few percentage points below ResNets of comparable size',
                    'Transformers lack some of the inductive biases'],
         column='full', above=4, end='Transformers lack some of the inductive biases'),
    dict(name='p1-intro-midsize2', page=1, anchor='inherent to CNNs, such as translation equivariance and locality',
         highlight=['translation equivariance and locality', 'do not generalize well when trained on insufficient amounts of data'],
         column='full', above=4, end='insufficient amounts of data.'),
    dict(name='p1-intro-scale', page=1, anchor='However, the picture changes if the models are trained on larger datasets',
         highlight=['14M-300M images', 'large scale training trumps inductive bias', 'public ImageNet-21k dataset or the in-house JFT-300M dataset',
                    '88.55%', '90.72%', '94.55%', '77.63%'],
         column='full', above=4, end='VTAB suite of 19 tasks.'),
    dict(name='p1-footnote', page=0, anchor='code and pre-trained models are available at',
         highlight=['1Fine-tuning code and pre-trained models are available at'], column='full', above=4,
         end='google-research/vision_transformer'),
    dict(name='p1-figure1', page=2, anchor='Figure 1: Model overview.', highlight=['split an image into fixed-size patches, linearly embed each of them'],
         figure=True, column='full', below=0),
    dict(name='p1-sec43-small', page=5, anchor='First, we pre-train ViT models on datasets of increasing size',
         highlight=['datasets of increasing size: ImageNet, ImageNet-21k, and JFT-300M', 'ViT-Large models underperform compared to ViT-Base models'],
         column='full', above=4, end='despite'),
    dict(name='p1-sec43-overtake', page=6, anchor='The BiT CNNs outperform ViT on ImageNet',
         highlight=['The BiT CNNs outperform ViT on ImageNet, but with the larger datasets, ViT overtakes'], column='full', above=4,
         end='ViT overtakes.'),
    dict(name='p1-table5', page=14, anchor='ViT-B/16', highlight=['77.91 73.38 76.53 71.16', '83.97 81.28 85.15 80.99 85.13', '84.15 80.73 87.12 84.37 88.04'],
         column='full', above=10, end='used to achieve results in Table 2.'),
    dict(name='p1-gpt3-abstract', arxiv_id='2005.14165', page=0, anchor='Specifically, we train GPT-3, an autoregressive language model',
         highlight=['175 billion parameters, 10x more than any previous non-sparse language model'], column='full', above=4, end='non-sparse language model'),
    dict(name='p1-gshard-abstract', arxiv_id='2006.16668', page=0, anchor='GShard enabled us to scale up multilingual neural machine translation',
         highlight=['beyond 600 billion parameters'], column='full', above=4, end='using automatic sharding.'),
    dict(name='p1-resnet-abstract', arxiv_id='1512.03385', page=0, anchor='On the ImageNet dataset we evaluate residual nets',
         highlight=['depth of up to 152 layers', '3.57% error on the ImageNet test set', '1st place on the ILSVRC 2015 classification task'],
         above=4, end='ILSVRC 2015 classification task.'),
    dict(name='p1-vaswani-abstract', arxiv_id='1706.03762', page=0, anchor='The dominant sequence transduction models',
         highlight=['based solely on attention mechanisms, dispensing with recurrence and convolutions entirely'],
         column='full', above=4, end='convolutions entirely.'),
    dict(name='p1-bert-abstract', arxiv_id='1810.04805', page=0, anchor='We introduce a new language representa',
         highlight=['pre-train deep bidirectional representations from unlabeled text', 'fine-tuned with just one additional output layer'],
         above=4, end='without substantial task-specific architecture modifications'),
]


def trim_stamp(jobs, dpi=190):
    """The rotated arXiv stamp on page 1 of some PDFs fools the margin logic of paper_shots (a short rotated word such
    as "3" passes its filter), so the crop starts at the page edge and the stamp shows. Cut the PNG back to the text
    column: the text's left edge minus 10 pt and right edge plus 10 pt, measured on the PDF page itself."""
    import os
    import pymupdf as fitz
    from PIL import Image
    from paper_shots import pdf, OUT
    for j in jobs:
        pg = pdf(j.get('arxiv_id', '2010.11929'))[j['page']]
        W = pg.rect.width
        ws = [w for w in pg.get_text('words') if (w[3] - w[1]) < 3 * (w[2] - w[0])]       # the same filter paper_shots uses
        crop_x0 = max(20, min(w[0] for w in ws) - 10)
        body = [w for w in ws if w[0] > 60 and w[2] < W - 40]                             # words of the text column, not the stamp
        tx0, tx1 = min(w[0] for w in body) - 10, max(w[2] for w in body) + 10
        cut = round((tx0 - crop_x0) * dpi / 72)
        keep = round((tx1 - crop_x0) * dpi / 72)
        p = os.path.join(OUT, j['name'] + '.png')
        im = Image.open(p)
        if cut < 8 or im.width <= keep - cut + 4:                                          # nothing to cut, or already trimmed
            continue
        im.crop((cut, 0, min(keep, im.width), im.height)).save(p)
        print(f"{j['name']}: trimmed {cut} px of margin on the left -> {min(keep, im.width) - cut}x{im.height}")


if __name__ == '__main__':
    run(JOBS)
    only = os.environ.get('ONLY')
    trim_stamp([j for j in JOBS if j.get('column') == 'full' and j['page'] == 0 and (not only or j['name'] in only.split(','))])
