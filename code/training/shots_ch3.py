"""Highlighted excerpts for Chapter 3 (how we measure a model): benchmark papers (MMLU, GSM8K, HumanEval, TruthfulQA,
HellaSwag, SuperGLUE, MMLU-Pro, GPQA), contamination (GPT-3), human and model judges (Chatbot Arena, MT-Bench,
length-controlled AlpacaEval), reward-model evaluation (RewardBench, Gao et al.) and the Qwen2.5 report.
Usage: python shots_ch3.py   (or ONLY=name1,name2 python shots_ch3.py)"""
import os
from paper_shots import run, pdf, find, YELLOW, OUT
import pymupdf as fitz


def manual(name, arxiv_id, page, rect, highlight=(), dpi=190, hl_nth=None):
    """A fixed crop rectangle (points) for excerpts the automatic crop cannot isolate."""
    pg = pdf(arxiv_id)[page]
    for phrase in highlight:
        rects = find(pg, phrase, (hl_nth or {}).get(phrase, 0))
        if not rects:
            raise SystemExit(f'{name}: highlight not found: {phrase!r}')
        for r in rects:
            pg.add_highlight_annot(r).set_colors(stroke=YELLOW)
    pix = pg.get_pixmap(clip=fitz.Rect(*rect), dpi=dpi, annots=True)
    pix.save(os.path.join(OUT, name + '.png'))
    print(f'{name}: page {page + 1}, {pix.width}x{pix.height} (manual crop)')


MANUAL = [
    dict(name='ch3-arena-bt', arxiv_id='2403.04132', page=3, rect=(303, 381, 547, 572),
         highlight=['so-called BT coefficients', 'binary cross-entropy']),
    dict(name='ch3-lc-eq1', arxiv_id='2404.04475', page=3, rect=(104, 574, 512, 708),
         highlight=['logistic regression that has 3 terms: model, length, and instruction']),
    dict(name='ch3-gao-fig1a', arxiv_id='2210.10760', page=2, rect=(100, 110, 512, 364)),
    dict(name='ch3-humaneval-fig3', arxiv_id='2107.03374', page=2, rect=(300, 488, 548, 609),
         highlight=['A numerically stable script for calculating an unbiased']),
]

JOBS = [
    # ---- benchmarks
    dict(name='ch3-mmlu-fig1', arxiv_id='2009.03300', page=1, figure=True, column='full',
         anchor='(a) An example of few-shot learning and inference', highlight=['Answer: C'], below=40),
    dict(name='ch3-mmlu-method', arxiv_id='2009.03300', page=5,
         anchor='For few-shot evaluation, we add up to 5 demonstration examples',
         highlight=['All prompts end with “Answer: ”', 'we treat the highest probability option as the prediction'],
         above=4, end='5 ﬁxed few-shot examples for each subject.'),
    dict(name='ch3-gsm8k-fig1', arxiv_id='2110.14168', page=1, figure=True,
         anchor='Figure 1: Three example problems from GSM8K', highlight=['Calculation annotations are highlighted in red'], below=0),
    dict(name='ch3-humaneval-eq1', arxiv_id='2107.03374', page=2,
         anchor='passes the unit tests, and the total fraction of problems solved is reported',
         highlight=['calculate the unbiased estimator'], above=4, end='numerically stable numpy implementation'),
    dict(name='ch3-humaneval-fig2', arxiv_id='2107.03374', page=2, figure=True,
         anchor='Figure 2. Three example problems from the HumanEval dataset',
         highlight=['Three example problems from the HumanEval dataset'], below=0),
    dict(name='ch3-truthfulqa-fig2', arxiv_id='2109.07958', page=2, figure=True, column='full',
         anchor='Figure 2: Larger models are less truthful', highlight=['Larger models are less truthful'], below=0),
    dict(name='ch3-hellaswag-abstract', arxiv_id='1905.07830', page=0,
         anchor='a new challenge dataset', highlight=['trivial for humans', 'state-of-the-art models struggle'],
         above=60, end='Adversarial Filtering (AF)'),
    dict(name='ch3-mmlupro-abstract', arxiv_id='2406.01574', page=0,
         anchor='expanding the choice set from four to ten options',
         highlight=['expanding the choice set from four to ten options',
                    'the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro'],
         above=40, end='just 2% in MMLU-Pro.'),
    dict(name='ch3-gpqa-abstract', arxiv_id='2311.12022', page=0,
         anchor='We present GPQA, a challenging dataset of 448 multiple-choice questions',
         highlight=['reach 65% accuracy', 'only reach 34% accuracy', 'unrestricted access to the web'],
         above=4, end='the questions are “Google-proof”'),
    dict(name='ch3-sclar-abstract', arxiv_id='2310.11324', page=0,
         anchor='extremely sensitive to subtle changes in prompt formatting',
         highlight=['performance differences of up to 76 accuracy points'], above=30, below=14),
    # ---- contamination
    dict(name='ch3-gpt3-contam', arxiv_id='2005.14165', page=30,
         anchor='Unfortunately, a bug resulted in only partial removal', highlight=['a bug resulted in only partial removal',
                                                                                   '13-gram overlap with anything in the pretraining set'],
         above=40, end='suggests that contamination, even if present'),
    dict(name='ch3-gpt3-fig42', arxiv_id='2005.14165', page=31, figure=True, column='full',
         anchor='Figure 4.2: Benchmark contamination analysis', highlight=['Benchmark contamination analysis'], below=0),
    dict(name='ch3-qwen25-decontam', arxiv_id='2412.15115', page=6,
         anchor='To prevent test data leakage', highlight=['we exclude potentially contaminated data using n-gram matching'],
         above=4, end='min(|st|, |se|).'),
    dict(name='ch3-rephrase-abstract', arxiv_id='2311.04850', page=0,
         anchor='While most data decontamination efforts apply string', highlight=['simple variations of test data (e.g., paraphras- ing, translation) can easily bypass these decon- tamination measures'],
         above=4, end='on par with GPT-4.'),
    # ---- the models we test, as reported
    dict(name='ch3-qwen25-table5', arxiv_id='2412.15115', page=9, column='full',
         anchor='Table 5: Performance of the smaller base models', highlight=['47.5', '41.6', '30.5'],
         hl_nth={}, above=4, end='Multi-Translation'),
    # ---- human and model judges
    dict(name='ch3-mtbench-types', arxiv_id='2306.05685', page=3,
         anchor='3.1 Types of LLM-as-a-Judge', highlight=['Pairwise comparison', 'Single answer grading', 'Reference-guided grading'],
         above=4, end='if the judge model changes.'),
    dict(name='ch3-mtbench-verbosity', arxiv_id='2306.05685', page=4,
         anchor='Verbosity bias is when an LLM judge favors longer', highlight=['Verbosity bias', '“repetitive list” attack'],
         above=4, end='cannot pass the more advanced “repetitive list” attack.'),
    # ---- reward models
    dict(name='ch3-rewardbench-fig1', arxiv_id='2403.13787', page=3, figure=True, column='full',
         anchor='Figure 1: The scoring method of the REWARDBENCH evaluation suite', highlight=['The scoring method'], below=0),
]

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    for m in MANUAL:
        if not only or m['name'] in only.split(','):
            manual(**m)
    run(JOBS)
