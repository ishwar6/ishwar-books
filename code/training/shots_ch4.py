"""Highlighted excerpts for Chapter 4 (pretraining): FineWeb, the Pile, BPE, Kaplan, Chinchilla, GPT-3, Llama 3,
mixed precision, deduplication and TinyStories."""
import os
import paper_shots
from paper_shots import run

JOBS = [
    dict(name='ch4-fineweb-base-filtering', arxiv_id='2406.17557',
         anchor='As a starting point to our filtering, we applied a basic filtering pipeline',
         highlight=['URL filtering using a blocklist', 'keep only English text with a score >= 0.65',
                    'quality and repetition filters from MassiveText'], above=4, end='the GPT-2 tokenizer.'),
    dict(name='ch4-fineweb-minhash', arxiv_id='2406.17557',
         anchor='Following RefinedWeb [50], we experimented with MinHash',
         highlight=['computed MinHashes using 112 hash functions in total', 'split into 14 buckets of 8 hashes each',
                    'targeting documents that are at least 75% similar'], above=4, end='considered duplicates of each other.'),
    dict(name='ch4-fineweb-custom-filters', arxiv_id='2406.17557',
         anchor='significant improvements on the aggregate bench',
         highlight=['documents where the fraction of lines ending with', 'fraction of charac', 'fraction of lines shorter than 30 characters is >= 0.67'],
         above=4, end='the aggregate score increased by about 1% in the'),
    dict(name='ch4-fineweb-edu', arxiv_id='2406.17557',
         anchor='To build the synthetic annotations, we use Llama-3-70B-Instruct',
         highlight=['score 460,000 randomly sampled', 'on a scale from 0 to 5', 'minimum threshold of 3'],
         above=4, end='Applying the classifier to the 15 trillion tokens of FineWeb required 6,000 H100 GPU hours.'),
    dict(name='ch4-fineweb-fig-pipeline', arxiv_id='2406.17557', figure=True,
         anchor='Figure 9:', highlight=['Figure 9:'], below=0),
    dict(name='ch4-pile-treemap', arxiv_id='2101.00027', page=1, figure=True, column='full',
         anchor='Figure 1: Treemap of Pile components by effective size.', highlight=['Treemap of Pile components'], below=0),
    dict(name='ch4-bpe-algorithm', arxiv_id='1508.07909', page=3, column='full',
         anchor='Algorithm 1 Learn BPE operations', highlight=['Algorithm 1 Learn BPE operations', 'best = max(pairs, key=pairs.get)'],
         above=4, end='print(best)'),
    dict(name='ch4-kaplan-fig1', arxiv_id='2001.08361', page=2, figure=True, column='full',
         anchor='Language modeling performance improves smoothly as we increase the model size',
         highlight=['For optimal performance all three factors must be scaled up in tandem'], below=0),
    dict(name='ch4-kaplan-6n', arxiv_id='2001.08361', page=6,
         anchor='compute as C ≈6N ﬂoating point operators per training token.',
         highlight=['C ≈6N ﬂoating point operators per training token'], above=60, below=6),
    dict(name='ch4-chinchilla-fig1', arxiv_id='2203.15556', page=1, figure=True, column='full',
         anchor='Figure 1 | Overlaid predictions.', highlight=['Overlaid predictions.'], below=0),
    dict(name='ch4-chinchilla-fig3', arxiv_id='2203.15556', page=5, figure=True, column='full',
         anchor='Figure 3 | IsoFLOP curves.', highlight=['IsoFLOP curves.'], below=40),
    dict(name='ch4-chinchilla-table3', arxiv_id='2203.15556', page=7,
         anchor='Table 3 | Estimated optimal training FLOPs and training tokens', highlight=['Estimated optimal training FLOPs and training tokens'],
         column='full', above=4, below=150),
    dict(name='ch4-chinchilla-fit', arxiv_id='2203.15556', page=24,
         anchor='406.4', highlight=['= 1.69', '= 406.4', '= 410.7'], column='full', above=70, below=8),
    dict(name='ch4-gpt3-fig1-2', arxiv_id='2005.14165', page=3, figure=True, column='full',
         anchor='Figure 1.2: Larger models make increasingly efficient use of in-context information.',
         highlight=['Larger models make increasingly efficient use of in-context information.'], below=0),
    dict(name='ch4-gpt3-fig2-1', arxiv_id='2005.14165', page=6, figure=True, column='full',
         anchor='Figure 2.1: Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning.',
         highlight=['Zero-shot, one-shot and few-shot'], below=0),
    dict(name='ch4-llama3-recipe', arxiv_id='2407.21783', page=13,
         anchor='We pre-train Llama 3 405B using AdamW with a peak learning rate',
         highlight=['peak learning rate of 8 × 10−5', 'linear warm up of 8,000', 'cosine learning rate schedule decaying to 8 × 10−7',
                    'initial batch size of 4M tokens', 'double the batch size again to 16M'], column='full', above=4,
         end='did not require interventions to correct for model training divergence.'),
    dict(name='ch4-llama3-mix-anneal', arxiv_id='2407.21783', page=5,
         anchor='Data mix summary. Our final data mix contains roughly 50% of tokens',
         highlight=['roughly 50% of tokens corresponding to general knowledge', '25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens',
                    'annealing improved the performance', 'by 24.0% and 6.4%, respectively'], column='full', above=4,
         end='However, the improvements on the 405B model are negligible'),
    dict(name='ch4-llama3-annealing', arxiv_id='2407.21783', page=14,
         anchor='During pre-training on the final 40M tokens, we linearly annealed the learning rate to 0',
         highlight=['linearly annealed the learning rate to 0', 'upsample data sources', 'average of model checkpoints'],
         column='full', above=4, end='to produce the final pre-trained model.'),
    dict(name='ch4-mixed-precision-fig1', arxiv_id='1710.03740', page=2, figure=True,
         anchor='Figure 1: Mixed precision training iteration for a layer.', highlight=['Mixed precision training iteration for a layer.'], below=0),
    dict(name='ch4-tinystories-abstract', arxiv_id='2305.07759', page=0,
         anchor='In this work, we introduce TinyStories, a synthetic dataset of short stories',
         highlight=['a synthetic dataset of short stories that only contain words that a typical 3 to 4-year-olds usually understand'],
         column='full', above=4, end='demonstrate reasoning capabilities.'),
    dict(name='ch4-dedup-abstract', arxiv_id='2107.06499', page=0,
         anchor='We find that existing language modeling datasets contain many near-duplicate',
         highlight=['over 1% of the unprompted output', 'a single 61 word English sentence that is repeated over 60,000 times',
                    'emit memorized text ten times less frequently'], above=4, end='thus allowing for more accurate evaluation.'),
]


def finish():
    from PIL import Image
    for f in sorted(os.listdir(paper_shots.OUT)):
        if f.startswith('ch4-') and f.endswith('.png'):
            path = os.path.join(paper_shots.OUT, f)
            im = Image.open(path)
            if f == 'ch4-fineweb-fig-pipeline.png' and im.height > 600:     # drop the sentence fragment above the plots
                im = im.crop((0, 80, im.width, im.height))
            if f == 'ch4-fineweb-custom-filters.png' and im.width > 1000:   # keep the text column, not the figure beside it
                im = im.crop((0, 0, int(im.width * 0.505), im.height))
            if f == 'ch4-tinystories-abstract.png' and im.width > 1300:     # trim the rotated arXiv stamp
                im = im.crop((95, 0, im.width, im.height))
            if f == 'ch4-bpe-algorithm.png' and im.width > 1000:            # keep only the left column (the code listing)
                im = im.crop((0, 0, int(im.width * 0.485), im.height))
            if im.width > 1100:
                im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
            im.save(path, optimize=True)


if __name__ == '__main__':
    import sys
    ok = []
    for job in JOBS:
        if os.environ.get('ONLY') and job['name'] not in os.environ['ONLY'].split(','):
            continue
        try:
            paper_shots.shot(**job); ok.append(job['name'])
        except SystemExit as e:
            print('FAILED', e)
    finish()
