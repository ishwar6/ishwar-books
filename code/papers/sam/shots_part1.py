"""Highlighted excerpts of the SAM paper for Part 1 (title, Figure 1, abstract, Section 1, Figure 2),
plus the lines of GPT-3, CLIP and the foundation-models report that Section 1 builds on."""
import os
from paper_shots import run

JOBS = [
    dict(name='p1-title', page=0, anchor='Segment Anything', highlight=['Segment Anything'], column='full', above=4,
         end='Segment Anything'),
    dict(name='p1-figure1', page=0, anchor='Figure 1: We aim to build a foundation model for segmentation',
         highlight=['promptable segmentation task', 'a segmentation model (SAM)', 'a data engine for collecting SA-1B'],
         figure=True, column='full', below=0),
    dict(name='p1-abstract-1', page=0, anchor='We introduce the Segment Anything (SA) project',
         highlight=['a new task, model, and dataset for image segmentation', 'in a data collection loop',
                    'largest segmentation dataset to date (by far)', 'over 1 billion masks on 11M licensed and privacy respecting images'],
         above=4, end='privacy respecting images.'),
    dict(name='p1-abstract-2', page=0, anchor='The model is designed and trained to be promptable',
         highlight=['designed and trained to be promptable', 'transfer zero-shot to new image distributions and tasks',
                    'competitive with or even superior to prior fully supervised results', 'Segment Anything Model (SAM)', '(SA-1B)'],
         above=4, end='computer vision.'),
    dict(name='p1-intro-llm-1', page=0, anchor='Large language models pre-trained on web-scale datasets',
         highlight=['strong zero-shot and few-shot generalization', 'foundation models', 'prompt engineering'],
         above=4, below=110),
    dict(name='p1-intro-llm-2', page=0, anchor='matching in some cases)',
         highlight=['model scale, dataset size, and total training compute'], above=4, end='training compute [56, 10, 21, 51].'),
    dict(name='p1-intro-clip', page=0, anchor='Foundation models have also been explored in computer vision',
         highlight=['CLIP [82] and ALIGN [55] use contrastive learning', 'zero-shot generalization to novel visual concepts',
                    'abundant training data does not exist'],
         above=4, end='abundant training data does not exist.'),
    dict(name='p1-intro-goal', page=0, anchor='In this work, our goal is to build a foundation model',
         highlight=['build a foundation model for image segmentation', 'promptable model', 'task, model, and data',
                    'What task will enable zero-shot generalization?', 'What is the corresponding model architecture?',
                    'What data can power this task and model?'],
         above=4, end='What data can power this task and model?'),
    dict(name='p1-intro-entangled', page=1, anchor='These questions are entangled',
         highlight=['promptable segmentation task', 'output segmentation masks in real-time', 'there is no web-scale data source for segmentation',
                    'data engine'],
         above=4, end='effectiveness of our approach.'),
    dict(name='p1-intro-task', page=1, anchor='In NLP and more recently computer vision, foundation models',
         highlight=['return a valid segmentation mask given any segmentation prompt', 'a point on a shirt may indicate either the shirt or the person wearing it',
                    'reasonable mask for at least one of those objects', 'both a pre-training objective'],
         above=14, end='via prompt engineering.'),
    dict(name='p1-intro-model', page=1, anchor='The promptable segmentation task and the goal of real-world use',
         highlight=['support flexible prompts', 'amortized real-time', 'ambiguity-aware', 'image encoder', 'prompt encoder',
                    'lightweight mask decoder', '50ms in a web browser', 'predict multiple masks for a single prompt'],
         above=14, end='shirt vs. person example.'),
    dict(name='p1-intro-data-1', page=1, anchor='To achieve strong generalization to new data distributions',
         highlight=['masks are not naturally abundant', 'model-in-the-loop dataset annotation', 'assisted-manual, semi-automatic, and fully automatic'],
         above=14, end='annotating the remaining objects,'),
    dict(name='p1-intro-data-2', page=1, anchor='helping increase mask diversity',
         highlight=['regular grid of foreground points', '100 high-quality masks per image'], above=4, end='masks per image.'),
    dict(name='p1-intro-dataset', page=1, anchor='Our final dataset, SA-1B, includes more than 1B masks',
         highlight=['more than 1B masks from 11M licensed and privacy-preserving images', 'collected fully automatically',
                    '400× more masks than any existing segmentation dataset'],
         above=14, end='new foundation models.'),
    dict(name='p1-intro-rai', page=1, anchor='We study and report on potential fairness concerns',
         highlight=['geographically and economically diverse', 'SAM performs similarly across different groups of people'],
         above=14, end='in the appendix.'),
    dict(name='p1-intro-exp', page=1, anchor='We extensively evaluate SAM.',
         highlight=['23 segmentation datasets', 'from a single foreground point', 'edge detection, object proposal generation, instance segmentation',
                    'text-to-mask prediction'],
         above=14, end='as we discuss in §8.'),
    dict(name='p1-intro-release', page=1, anchor='We are releasing the SA-1B dataset for research purposes',
         highlight=['permissive open license (Apache 2.0)'], above=14, end='with an online demo.'),
    dict(name='p1-figure2', page=2, anchor='Figure 2: Example images with overlaid masks', highlight=['1.1B high-quality segmentation masks',
         'annotated fully automatically by SAM'], figure=True, column='full', below=0),
    # the sources Section 1 builds on
    dict(name='p1-gpt3-abstract', arxiv_id='2005.14165', page=0, anchor='Specifically, we train GPT-3, an autoregressive language model',
         highlight=['175 billion parameters', 'without any gradient updates or fine-tuning', 'specified purely via text interaction with the model'],
         column='full', above=4, end='text interaction with the model.'),
    dict(name='p1-clip-abstract', arxiv_id='2103.00020', page=0, anchor='We demonstrate that the simple pre-training task of predicting',
         highlight=['predicting which caption goes with which image', '400 million (image, text) pairs',
                    'natural language is used to reference learned visual concepts', 'zero-shot transfer of the model to downstream tasks'],
         above=4, end='zero-shot transfer of the model to downstream tasks.'),
    dict(name='p1-foundation-def', arxiv_id='2108.07258', page=2, anchor='trained on broad data',
         highlight=['trained on broad data (generally using self-supervision at scale) that can be adapted'], above=20, below=20),
]


def trim_stamp(jobs, dpi=190):
    """The rotated arXiv stamp on page 1 can sit inside a full-width crop. Cut the PNG back to the text column:
    the text's left edge minus 10 pt and right edge plus 10 pt, measured on the PDF page itself."""
    import pymupdf as fitz
    from PIL import Image
    from paper_shots import pdf, OUT
    for j in jobs:
        pg = pdf(j.get('arxiv_id', '2304.02643'))[j['page']]
        W = pg.rect.width
        ws = [w for w in pg.get_text('words') if (w[3] - w[1]) < 3 * (w[2] - w[0])]
        crop_x0 = max(20, min(w[0] for w in ws) - 10)
        body = [w for w in ws if w[0] > 60 and w[2] < W - 40]
        tx0, tx1 = min(w[0] for w in body) - 10, max(w[2] for w in body) + 10
        cut = round((tx0 - crop_x0) * dpi / 72)
        keep = round((tx1 - crop_x0) * dpi / 72)
        p = os.path.join(OUT, j['name'] + '.png')
        im = Image.open(p)
        if cut < 8 or im.width <= keep - cut + 4:
            continue
        im.crop((cut, 0, min(keep, im.width), im.height)).save(p)
        print(f"{j['name']}: trimmed {cut} px of margin on the left -> {min(keep, im.width) - cut}x{im.height}")


if __name__ == '__main__':
    run(JOBS)
    only = os.environ.get('ONLY')
    trim_stamp([j for j in JOBS if j.get('column') == 'full' and j['page'] == 0 and 'arxiv_id' not in j
                and (not only or j['name'] in only.split(','))])
