"""Highlighted excerpts for Part 6 (impact, limits and summary): Section 8 and the conclusion of the SAM paper,
and five later papers that answer the limits:
SAM 2, MobileSAM, HQ-SAM, MedSAM and SAM 3."""
import os
from paper_shots import run, OUT

SAM2, MOBILE, HQ, MED, SAM3 = '2408.00714', '2306.14289', '2306.01567', '2304.12306', '2511.16719'

JOBS = [
    # ---------------------------------------------------------------- page 12: Section 8 and the conclusion
    dict(name='p6-foundation', page=11, anchor='Foundation models. Pre-trained models have been adapted',
         highlight=['trained on broad data at scale and are adaptable to a wide range of downstream tasks',
                    'inherently limited scope', 'the vast majority of its capabilities come from large-scale supervised training'],
         above=4, end='provides an effective solution.'),
    dict(name='p6-composition', page=11, anchor='Compositionality. Pre-trained models can power new',
         highlight=['make this kind of composition straightforward', 'reliable interface between SAM and other components',
                    'prompted with gaze points detected by a wearable device', 'without need for additional training'],
         above=4, end='without need for additional training.'),
    dict(name='p6-limits-1', page=11, anchor='Limitations. While SAM performs well in general',
         highlight=['It can miss fine structures', 'hallucinates small disconnected components at times',
                    'does not produce boundaries as crisply as more computationally intensive methods that “zoom-in”'],
         above=4, end='methods that “zoom-in”, e.g. [18].'),
    dict(name='p6-limits-2', page=11, anchor='In general, we expect dedicated interactive segmentation',
         highlight=['outperform SAM when many points are provided', 'designed for generality and breadth of use rather than high IoU interactive segmentation'],
         above=4, end='interactive segmentation. More'),
    dict(name='p6-limits-3', page=11, anchor='over, SAM can process prompts in real-time',
         highlight=['not real-time when using a heavy image encoder'], above=4, end='a heavy image encoder.'),
    dict(name='p6-limits-4', page=11, anchor='Our foray into the text-to-mask task',
         highlight=['exploratory and not entirely robust', 'unclear how to design simple prompts that implement semantic and panoptic segmentation',
                    'domain-specific tools'],
         above=4, end='in their respective domains.'),
    dict(name='p6-conclusion', page=11, anchor='Conclusion. The Segment Anything project',
         highlight=['lift image segmentation into the era of foundation models', 'a new task (promptable segmentation), model (SAM), and dataset (SA-1B)',
                    'remains to be seen by how it is used in the community'],
         above=4, end='help pave the path ahead.'),
    # ---------------------------------------------------------------- the papers that came after
    dict(name='p6-sam2-abstract', arxiv_id=SAM2, page=0, anchor='We present Segment Anything Model 2',
         highlight=['promptable visual segmentation in images and videos', 'streaming memory for real-time video processing',
                    'more accurate and 6× faster than the Segment Anything Model (SAM)'],
         above=4, end='code for model training and our demo.'),
    dict(name='p6-sam2-memory', arxiv_id=SAM2, page=4, anchor='Memory bank. The memory bank retains information',
         highlight=['FIFO queue of memories of up to N recent frames', 'up to M prompted frames', 'object pointers'],
         above=4, end='these object pointers.'),
    dict(name='p6-sam2-image', arxiv_id=SAM2, page=9, anchor='The first column (SA-23 All) shows accuracy',
         highlight=['58.9 mIoU with 1 click', '58.1 mIoU with 1 click', 'smaller but more effective Hiera image encoder'],
         above=4, end='smaller but more effective Hiera image encoder in SAM 2.'),
    dict(name='p6-mobilesam-abstract', arxiv_id=MOBILE, page=0, anchor='Many of such applications need to be run',
         highlight=['replacing the heavyweight image encoder with a lightweight one', 'decoupled distillation',
                    'single GPU within less than one day', 'more than 60 times smaller'],
         above=4, end='4ms on the mask decoder.'),
    dict(name='p6-mobilesam-table', arxiv_id=MOBILE, page=4, anchor='Table 3: Comparison of the parameters and speed for the image encoder',
         highlight=['632M', '5.78M', '452ms', '8ms'], above=4, below=95),
    dict(name='p6-mobilesam-eval', arxiv_id=MOBILE, page=5, anchor='of MobileSAM, however, we omit this step',
         highlight=['we compute the mIoU between the masks predicted by the original SAM and our MobileSAM'],
         above=4, end='our MobileSAM.'),
    dict(name='p6-hq-problem', arxiv_id=HQ, page=0, anchor='While SAM has achieved impressive performance',
         highlight=['Coarse mask boundaries, often even neglecting the segmentation of thin object structures',
                    'Incorrect predictions, broken masks'],
         above=4, end='rightmost column of Figure 1.'),
    dict(name='p6-hq-token', arxiv_id=HQ, page=0, anchor='We design a learnable High-Quality Output',
         highlight=['learnable High-Quality Output Token', 'early and final ViT features', '44K fine-grained masks', '4 hours on 8 GPUs'],
         above=4, end='4 hours on 8 GPUs.'),
    dict(name='p6-medsam-abstract', arxiv_id=MED, page=0, anchor='Here we present MedSAM',
         highlight=['1,570,263 image-mask pairs', '10 imaging modalities', '86 internal validation tasks and 60 external validation tasks'],
         above=4, end='modality-wise specialist models.'),
    dict(name='p6-medsam-train', arxiv_id=MED, page=11, anchor='The model was initialized with the pre-trained SAM',
         highlight=['ViT-Base', '89,670,912 and 4,058,340', 'random perturbation of 0-20 pixels'],
         above=4, end='perturbation of 0-20 pixels.'),
    dict(name='p6-medsam-limits', arxiv_id=MED, page=8, anchor='While MedSAM boasts strong capabilities',
         highlight=['modality imbalance in the training set', 'vessel-like branching structures', 'arteries and veins share the same bounding box'],
         above=4, end='share the same bounding box in eye fundus images.'),
    dict(name='p6-sam3-abstract', arxiv_id=SAM3, page=0, anchor='We present Segment Anything Model (SAM) 3',
         highlight=['concept prompts', 'short noun phrases', 'all matching object instances', '4M unique concept labels'],
         above=4, end='(SA-Co) benchmark for promptable'),
    dict(name='p6-sam3-limits', arxiv_id=SAM3, page=17, anchor='SAM 3 struggles to generalize to fine-grained out-of-domain concepts',
         highlight=['fine-grained out-of-domain concepts', 'inherently more challenging than the class-agnostic generalization'],
         above=4, end='diverse settings.'),
]

# arXiv prints its identifier as a rotated stamp in the left margin; cut it off (pixels from the left edge, at 190 dpi)
TRIM_LEFT = {'p6-sam2-abstract': 130, 'p6-mobilesam-abstract': 215, 'p6-mobilesam-table': 495, 'p6-hq-token': 305,
             'p6-hq-problem': 215, 'p6-medsam-abstract': 255, 'p6-sam3-abstract': 160}


def trim(name, left):
    from PIL import Image
    path = os.path.join(OUT, name + '.png')
    im = Image.open(path)
    im.crop((left, 0, im.width, im.height)).save(path)
    print(f'{name}: trimmed {left} px on the left')


if __name__ == '__main__':
    run(JOBS)
    for n, left in TRIM_LEFT.items():
        if not os.environ.get('ONLY') or n in os.environ.get('ONLY').split(','):
            trim(n, left)
