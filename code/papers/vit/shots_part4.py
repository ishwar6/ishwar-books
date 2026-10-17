"""Highlighted excerpts of the ViT paper for Part 4 (Section 4.2, Table 2, Figure 2, Section 4.3, Figures 3 and 4,
Appendix C with Table 5, Appendix D.1 with Table 7, D.9 and D.10 with Table 9) and of the papers it compares with or
evaluates on (Big Transfer, Noisy Student, VTAB, ImageNet ReaL, ObjectNet)."""
from paper_shots import run

JOBS = [
    # ---- Section 4.2, Comparison to state of the art (page index 4, bottom; page index 5)
    dict(name='p4-sota-intro', page=4, anchor='We first compare our largest models',
         highlight=['ViT-H/14 and ViT-L/16', 'state-of-the-art CNNs', 'Big Transfer (BiT)', 'supervised transfer learning with large ResNets',
                    'Noisy Student', 'semi-supervised learning on ImageNet and JFT-', 'state of the art on ImageNet'],
         end='300M with the labels removed. Currently,', above=30),
    dict(name='p4-core-days', page=4, anchor='All models were trained on TPUv3 hardware',
         highlight=['TPUv3-core-days', 'number of TPU', 'v3 cores (2 per chip) used for training multiplied by the training time in days'],
         end='multiplied by the training time in days.', above=4),
    dict(name='p4-table2', page=5, anchor='Table 2:',
         highlight=['Comparison with state of the art on popular image classification benchmarks',
                    'averaged over three fine-tuning runs', 'substantially less computational resources to pre-train'],
         figure=True, column='full', below=0, fig_top=128),
    dict(name='p4-table2-text', page=4, anchor='Table 2 shows the results.',
         highlight=['outperforms BiT-L', 'on all tasks', 'substantially less computa', 'tional resources to train',
                    'especially', 'on the more challenging datasets'], end='Interestingly, this', above=4),
    dict(name='p4-table2-text2', page=5, anchor='model still took substantially less compute to pre-train',
         highlight=['substantially less compute to pre-train than prior state of the art', 'pre-training efficiency may be affected not only by the architecture choice',
                    'training schedule, optimizer, weight decay', 'public ImageNet-21k dataset', 'standard cloud TPUv3 with 8 cores',
                    'approximately 30 days'], end='proximately 30 days.', above=4),
    dict(name='p4-figure2', page=5, anchor='Figure 2: Breakdown of VTAB performance',
         highlight=['Breakdown of VTAB performance'], figure=True, column='full', below=0),
    dict(name='p4-figure2-text', page=5, anchor='Figure 2 decomposes the VTAB tasks',
         highlight=['VIVI', 'S4L', 'outperforms BiT-R152x4, and other methods, on the Natural and Structured tasks',
                    'performance of the top two models is similar'], end='the top two models is similar.', above=4),
    # ---- Section 4.3, Pre-training data requirements (page index 5 bottom, page index 6)
    dict(name='p4-data-intro', page=5, anchor='The Vision Transformer performs well when pre-trained on a large JFT-300M dataset.',
         highlight=['how crucial is the dataset size', 'two series of', 'datasets of increasing size', 'weight decay, dropout, and label smoothing',
                    'ViT-Large models underperform compared to ViT-Base models', 'their performances are similar',
                    'full benefit of larger models'], end='Figure 3 also shows the performance', above=30),
    dict(name='p4-footnote2', page=5, anchor='the ImageNet pre-trained models are also fine-tuned',
         highlight=['also fine-tuned, but again on ImageNet', 'resolution increase during fine-tuning improves the performance'], above=6, below=14),
    dict(name='p4-figure3', page=6, anchor='Figure 3:',
         highlight=['Transfer to ImageNet.', 'shaded area', 'they shine when pre-trained on', 'larger ViT variants', 'overtake smaller ones as the dataset grows'],
         figure=True, below=0),
    dict(name='p4-figure3-text', page=6, anchor='region spanned by BiT models of different sizes.',
         highlight=['region spanned by BiT models of different sizes', 'The BiT CNNs outperform ViT on ImageNet', 'ViT overtakes'],
         end='with the larger datasets, ViT overtakes.', above=4),
    dict(name='p4-subsets', page=6, anchor='Second, we train our models on random subsets',
         highlight=['random subsets of 9M, 30M, and 90M', 'do not perform additional regularization', 'intrinsic model properties',
                    'early-stopping', 'few-shot linear accuracy', 'overfit more than ResNets', 'ViT-B/32 is slightly faster than',
                    'much worse on the 9M subset, but better on 90M+ subsets', 'convolutional inductive', 'learning the relevant patterns directly from',
                    'even beneficial'], end='data is sufficient, even beneficial.', above=4),
    dict(name='p4-figure4', page=6, anchor='Figure 4: Linear few-shot evaluation on Ima',
         highlight=['Linear few-shot evaluation on Ima', 'ResNets per', 'plateau sooner than ViT', 'ViT-b is ViT-B', 'with all hidden dimensions halved'],
         figure=True, below=0),
    dict(name='p4-outlook', page=6, anchor='Overall, the few-shot results on ImageNet',
         highlight=['seem promising for very low-data transfer', 'exciting direction of future work'], end='is an exciting direction of future work.', above=4),
    # ---- Appendix C, Table 5 (page index 13 bottom, page index 14)
    dict(name='p4-appc', page=13, anchor='We report detailed results corresponding to the figures',
         highlight=['Table 5 corresponds', 'to Figure 3', 'on datasets of increasing size'], end='ImageNet, ImageNet-21k, and JFT-300M. Table 6 corresponds to', above=26),
    dict(name='p4-table5', page=14, anchor='Table 5: Top1 accuracy',
         highlight=['These values correspond to Figure 3 in the main text', 'fine-tuned at 384 resolution', 'without additional',
                    'Polyak averaging and 512 resolution images'], figure=True, column='full', below=0, fig_top=234),
    # ---- Appendix D.1, SGD vs. Adam (page index 14 bottom, page index 15)
    dict(name='p4-d1', page=14, anchor='ResNets are typically trained with SGD',
         highlight=['typically trained with SGD', 'quite unconventional'], above=40, below=26),
    dict(name='p4-table7', page=15, anchor='Table 7: Fine-tuning ResNet models pre-trained with Adam and SGD.',
         highlight=['Fine-tuning ResNet models pre-trained with Adam and SGD'], figure=True, column='full', below=0, fig_top=104),
    dict(name='p4-d1-text', page=15, anchor='performance of two ResNets',
         highlight=['Adam pre-training outperforms SGD pre-training on most datasets and on average',
                    'we pre-train only', 'for 7 epochs, not 30'], end='for 7 epochs, not 30.', above=4),
    # ---- Appendix D.9 and D.10 (page index 19), Table 9 (page index 21)
    dict(name='p4-d9', page=19, anchor='OBJECTNET RESULTS',
         highlight=['ObjectNet benchmark', '82.1% top-5 accuracy and 61.7% top-1 accuracy'], end='61.7% top-1 accuracy.', above=6),
    dict(name='p4-table9', page=21, anchor='Table 9: Breakdown of VTAB-1k performance across tasks.',
         highlight=['Breakdown of VTAB-1k performance across tasks'], end='ViT-L/16 (I21k)', above=6, column='full'),
    # ---- the papers it compares with and evaluates on
    dict(name='p4-ext-bit', arxiv_id='1912.11370', page=0, anchor='Transfer of pre-trained representations improves sample',
         highlight=['pre-training on large su', 'pervised datasets and fine-tuning the model on a target task', 'from 1 example per class to 1 M total examples',
                    '87.5% top-1 accuracy on ILSVRC-2012', '76.3%', 'on the 19 task Visual Task Adaptation Benchmark (VTAB)'],
         end='analysis of the main components that lead to high transfer performance.', above=4),
    dict(name='p4-ext-bit-objectnet', arxiv_id='1912.11370', page=20, anchor='Table 6: Results (%) on the ObjectNet test set.',
         highlight=['Results (%) on the ObjectNet test set'], end='R152x4 36.0', above=6, column='full'),
    dict(name='p4-ext-noisy', arxiv_id='1911.04252', page=0, anchor='We present Noisy Student Training, a semi-supervised',
         highlight=['88.4% top', '1 accuracy on ImageNet', 'teacher to generate pseudo labels for', '300M unlabeled images',
                    'train a larger Efficient', 'Net as a student model'], end='so that the student generalizes better than the', above=4),
    dict(name='p4-ext-vtab', arxiv_id='1910.04867', page=0, anchor='Representation learning promises to unlock deep',
         highlight=['Visual Task Adaptation Benchmark (VTAB)', 'adapt', 'to diverse, unseen tasks with few examples'],
         end='close are we to general visual representations?', above=4),
    dict(name='p4-ext-real', arxiv_id='2006.07159', page=0, anchor='Yes, and no. We ask whether recent progress',
         highlight=['overfit to the idiosyncrasies of its labeling procedure', 'more robust procedure for collecting human annotations',
                    'gains to be substantially', 'smaller than those reported on the original labels'],
         end='future research in visual', above=4),
    dict(name='p4-ext-objectnet', arxiv_id='objectnet', page=0, anchor='We collect a large real-world test set, ObjectNet',
         highlight=['backgrounds, rotations, and imaging viewpoints are random', '40-45% drop in performance',
                    'does not', 'come paired with a training set'], end='harder, due to the controls.', above=4),
]

# Figures 3 and 4 sit side by side on one page, and two cited abstracts have arXiv's side stamp in the margin:
# keep only the part of the rendered PNG that belongs to the excerpt (left and right edges as fractions of the width).
TRIM = {'p4-figure3': (0.0, 0.50), 'p4-figure4': (0.485, 1.0), 'p4-ext-bit': (0.285, 1.0), 'p4-ext-real': (0.225, 1.0)}


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
        print(f'{name}: trimmed to {int((b - a) * im.width)} px wide')


if __name__ == '__main__':
    run(JOBS)
    trim()
