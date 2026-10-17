"""Highlighted excerpts of the SAM paper for Part 2 (Section 2, Figure 3), plus the two places that Section 2 points to:
the "Resolving ambiguity" paragraph of Section 3 and the training algorithm of Appendix A."""
import os
from paper_shots import run, OUT

JOBS = [
    dict(name='p2-intro', page=3, anchor='We take inspiration from NLP, where the next token',
         highlight=['next token prediction task', 'foundation model pre-training', 'solve diverse downstream tasks via prompt engineering',
                    'a task with analogous capabilities'],
         above=30, end='analogous capabilities.'),
    dict(name='p2-task-1', page=3, anchor='Task. We start by translating the idea of a prompt',
         highlight=['a set of foreground / background points', 'a rough box or mask', 'free-form text',
                    'any information indicating what to segment in an image', 'return a valid segmentation mask given any prompt'],
         above=4, end='mask given any prompt.'),
    dict(name='p2-task-2', page=3, anchor='The requirement of a',
         highlight=['a prompt is ambiguous and could refer to multiple objects', 'a reasonable mask for at least one of those objects',
                    'output a coherent response to an ambiguous prompt', 'a natural pre-training algorithm',
                    'a general method for zero-shot transfer'],
         above=4, end='tasks via prompting.'),
    dict(name='p2-fig3', page=3, anchor='Figure 3: Each column shows 3 valid masks',
         highlight=['3 valid masks', 'a single ambiguous point prompt'], figure=True, below=0),
    dict(name='p2-pretrain-1', page=3, anchor='Pre-training. The promptable segmentation task suggests',
         highlight=['simulates a sequence of prompts', 'compares the model’s mask predictions against the ground truth',
],
         above=4, end='against the ground truth.'),
    dict(name='p2-pretrain-2', page=3, anchor='We adapt this method from interactive',
         highlight=['eventually predict a valid mask after enough user input', 'always predict a valid mask for any prompt',
                    'automatic annotation', 'specialized modeling and training loss choices'],
         above=4, end='which we discuss in §3.'),
    dict(name='p2-zeroshot', page=3, anchor='Zero-shot transfer. Intuitively, our pre-training task',
         highlight=['respond appropriately to any prompt at inference time', 'engineering appropriate prompts',
                    'providing the detector’s box output as a prompt to our model', 'cast as prompting', 'five diverse example tasks'],
         above=4, end='in our experiments in §7.'),
    dict(name='p2-related-1', page=3, anchor='Related tasks. Segmentation is a broad field',
         highlight=['interactive segmentation', 'edge detection', 'object proposal generation', 'instance segmentation'],
         hl_nth={'interactive segmentation': 2, 'instance segmentation': 1},
         above=4, end='The goal of our promptable segmentation task is to produce'),
    dict(name='p2-related-2', page=3, anchor='a broadly capable model that can adapt to many',
         highlight=['many (though not all)', 'task generalization', 'a fixed set of tasks', 'the training and test tasks are the same',
                    'a new, different task at inference time', 'a component in a larger system'],
         above=4, end='with an existing object detector.'),
    dict(name='p2-discussion', page=3, anchor='Discussion. Prompting and composition are powerful tools',
         highlight=['tasks unknown at the time of model design', 'CLIP [82] is the text-image alignment component of the DALL·E [83]',
                    'composable system design', 'composed into a larger algorithmic system'],
         above=4, end='as we will demonstrate.'),
    # the places Section 2 points to
    dict(name='p2-resolving-a', page=4, anchor='Resolving ambiguity. With one output',
         highlight=['the model will average multiple valid masks', 'predict multiple output masks for a single prompt',
                    'whole, part, and subpart', 'backprop only the minimum'],
         above=4, below=90),
    dict(name='p2-resolving-b', page=4, anchor='loss [15, 45, 64] over masks', highlight=['estimated IoU'],
         above=4, end='estimated IoU) for each mask.'),
    dict(name='p2-algo-1', page=16, anchor='Training algorithm. Following recent approaches',
         highlight=['simulate an interactive segmentation setup during training', 'either a foreground point or bounding box',
                    'sampled uniformly from the ground truth mask', 'standard deviation equal to 10% of the box sidelength',
                    'maximum of 20 pixels'],
         above=4, end='may draw a loose box.'),
    dict(name='p2-algo-2', page=16, anchor='After making a prediction from this first prompt',
         highlight=['selected uniformly from the error region', 'false negative or false positive',
                    'unthresholded mask logits', 'the one with the highest predicted IoU', '11 total iterations'],
         above=4, end='refine its own mask predictions.'),
]

def stitch(parts, name):
    """Put excerpts from two columns one under the other, as one picture."""
    from PIL import Image
    ims = [Image.open(os.path.join(OUT, p + '.png')) for p in parts]
    out = Image.new('RGB', (max(i.width for i in ims), sum(i.height for i in ims)), (255, 255, 255))
    y = 0
    for i in ims:
        out.paste(i, (0, y)); y += i.height
    out.save(os.path.join(OUT, name + '.png'))
    for p in parts:
        os.remove(os.path.join(OUT, p + '.png'))
    print(f'{name}: {out.width}x{out.height} from {", ".join(parts)}')


if __name__ == '__main__':
    run(JOBS)
    if not os.environ.get('ONLY') or 'p2-resolving' in os.environ.get('ONLY'):
        stitch(['p2-resolving-a', 'p2-resolving-b'], 'p2-resolving')
