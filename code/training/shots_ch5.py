"""Highlighted excerpts for Chapter 5 (supervised fine-tuning): FLAN, T0, Super-NaturalInstructions, Self-Instruct, InstructGPT,
LIMA, Tulu 3, LoRA, QLoRA and "LoRA learns less and forgets less"."""
import os
import paper_shots

JOBS = [
    dict(name='ch5-flan-fig1', arxiv_id='2109.01652', page=0, figure=True, column='full',
         anchor='Figure 1: Top: overview of instruction tuning and FLAN.', highlight=['overview of instruction tuning and FLAN'], below=0),
    dict(name='ch5-flan-templates', arxiv_id='2109.01652', page=2, figure=True,
         anchor='Figure 4: Multiple instruction templates describing a natural language inference task.',
         highlight=['Multiple instruction templates'], below=0),
    dict(name='ch5-t0-fig1', arxiv_id='2110.08207', page=1, figure=True, column='full',
         anchor='Figure 1: Our model and prompt format.', highlight=['Our model and prompt format.'], below=0),
    dict(name='ch5-superni-fig1', arxiv_id='2204.07705', page=0, figure=True,
         anchor='Figure 1:', highlight=['Figure 1:'], below=0),
    dict(name='ch5-selfinstruct-fig2', arxiv_id='2212.10560', page=1, figure=True, column='full',
         anchor='Figure 2: A high-level overview of SELF-INSTRUCT.', highlight=['A high-level overview of SELF-INSTRUCT.'], below=0),
    dict(name='ch5-instructgpt-sft', arxiv_id='2203.02155', page=6, column='full',
         anchor='From these prompts, we produce three different datasets used in our',
         highlight=['our SFT dataset, with labeler demonstrations used to train our SFT models', 'The SFT dataset contains about 13k training'],
         above=4, end='and the PPO dataset has 31k training prompts (only from the API).'),
    dict(name='ch5-lima-abstract', arxiv_id='2305.11206', page=0, column='full',
         anchor='We measure the relative importance of these two stages by training LIMA',
         highlight=['only 1,000 carefully curated prompts and responses', 'almost all knowledge in large language models is learned during pretraining',
                    'only limited instruction tuning data is necessary'],
         above=4, end='to teach models to produce high quality output.'),
    dict(name='ch5-lima-hypothesis', arxiv_id='2305.11206', page=1, column='full',
         anchor='A model’s knowledge and capabilities are learnt',
         highlight=['knowledge and capabilities are learnt almost entirely during pretraining',
                    'alignment teaches it which subdistribution of formats should be used when interacting with users'],
         above=20, end='tune a pretrained language model with a rather small set of examples'),
    dict(name='ch5-tulu3-fig1', arxiv_id='2411.15124', page=4, figure=True, column='full',
         anchor='Figure 1 An overview of the Tülu 3 recipe.', highlight=['An overview of the Tülu 3 recipe.'], below=0),
    dict(name='ch5-tulu3-sftmix', arxiv_id='2411.15124', page=14, figure=True, column='full',
         anchor='Figure 2 The Tülu 3 final SFT mix by source and length', highlight=['The Tülu 3 final SFT mix by source'], below=0),
    dict(name='ch5-lora-fig1', arxiv_id='2106.09685', page=0, figure=True,
         anchor='Figure 1: Our reparametriza', highlight=['Our reparametriza'], below=0),
    dict(name='ch5-lora-eq', arxiv_id='2106.09685', page=3, column='full',
         anchor='For h = W0x, our modiﬁed forward pass yields:',
         highlight=['h = W0x + ∆Wx = W0x + BAx', 'random Gaussian initialization for A and', 'zero for B, so ∆W = BA is zero at the beginning of training'],
         above=60, end='roughly the same as tuning the learning'),
    dict(name='ch5-qlora-fig1', arxiv_id='2305.14314', page=2, figure=True, column='full',
         anchor='Figure 1: Different finetuning methods and their memory requirements.',
         highlight=['Different finetuning methods and their memory requirements.'], below=0),
    dict(name='ch5-qlora-abstract', arxiv_id='2305.14314', page=0, column='full',
         anchor='We present QLORA, an efficient finetuning approach',
         highlight=['finetune a 65B parameter model on a single 48GB GPU', 'frozen, 4-bit quantized pretrained language model into Low Rank', '4-bit NormalFloat (NF4)'],
         above=4, end='Paged Optimizers to manage memory spikes.'),
    dict(name='ch5-lora-forgets-fig3', arxiv_id='2405.09673', page=7, figure=True, column='full',
         anchor='Figure 3: LoRA vs. full finetuning tradeoff for Llama-2-7B.', highlight=['LoRA learns', 'less (lower values on the y-axis) and forgets less'], below=0),
]


def finish():
    from PIL import Image
    for f in sorted(os.listdir(paper_shots.OUT)):
        if f.startswith('ch5-') and f.endswith('.png') and not f.endswith('-run.png'):
            path = os.path.join(paper_shots.OUT, f)
            im = Image.open(path)
            if f == 'ch5-lora-fig1.png' and im.width > 1000:        # the figure sits in a narrow right-hand column
                im = im.crop((int(im.width * 0.755), 0, im.width, im.height))
            if f in ('ch5-flan-fig1.png', 'ch5-lima-abstract.png') and im.width > 1300:   # trim the rotated arXiv stamp
                im = im.crop((80, 0, im.width, im.height))
            if f == 'ch5-superni-fig1.png' and im.height > 1000:    # drop the author affiliations above the figure
                im = im.crop((0, 100, im.width, im.height))
            if im.width > 1100:
                im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
            im.save(path, optimize=True)


if __name__ == '__main__':
    for job in JOBS:
        if os.environ.get('ONLY') and job['name'] not in os.environ['ONLY'].split(','):
            continue
        try:
            paper_shots.shot(**job)
        except SystemExit as e:
            print('FAILED', e)
    finish()
