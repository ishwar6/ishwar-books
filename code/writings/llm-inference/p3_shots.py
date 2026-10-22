"""Highlighted paper excerpts for Part 3 of "LLM Inference from the Ground Up" (writings/llm-inference-3-vllm).
Uses the shared cutter in code/papers/bert/paper_shots.py; writes PNGs to public/img/writings/llm-inference/.
Paper: Kwon et al. (2023), arXiv 2309.06180 (pages are 0-based here). Sarathi-Serve: arXiv 2403.02310.
Run: python p3_shots.py   (or ONLY=p3-figure6,p3-kernel-overhead python p3_shots.py)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers', 'bert'))
import paper_shots
from paper_shots import shot
paper_shots.OUT = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'writings', 'llm-inference')

V = '2309.06180'
SARATHI = '2403.02310'
JOBS = [
    dict(name='p3-vm-idea', arxiv_id=V, page=4, anchor='memory manager is analogous',
         highlight=['analogous', 'to the virtual memory', 'fixed-sized pages', 'needs not to', 'be fully reserved in advance',
                    'fixed-size KV blocks, like pages in virtual memory'],
         above=4, end='fixed-size KV blocks, like pages in virtual memory.'),
    dict(name='p3-figure4', arxiv_id=V, page=4, anchor='Figure 4. vLLM system overview.', figure=True, column='full',
         below=0, highlight=['vLLM system overview.']),
    dict(name='p3-figure5', arxiv_id=V, column='full', page=4, anchor='Figure 5. Illustration of the PagedAttention algorithm,',
         figure=True, below=0, highlight=['non-contiguous blocks in the memory.']),
    dict(name='p3-kernel-fetch', arxiv_id=V, page=4, anchor='During the attention computation, the PagedAttention',
         highlight=['identifies and fetches different KV blocks separately', 'not contiguous on the physical memory'],
         above=4, end='vectors 𝑉𝑗in a block to derive the final attention output 𝑜𝑖.'),
    dict(name='p3-figure6', arxiv_id=V, column='full', page=5, anchor='Figure 6. Block table translation in vLLM.', figure=True,
         below=0, highlight=['Block table translation in vLLM.']),
    dict(name='p3-walkthrough', arxiv_id=V, page=5, anchor='Next, we walk through an example, as in Fig. 6',
         highlight=['does not require', 'reserving the memory for the maximum possible generated', 'The prompt has 7',
                    'tokens, so vLLM maps the first 2 logical KV blocks', 'one slot remains available',
                    'allocates a', 'new physical block (physical block 3)'],
         above=4, end='mapping in the block table.'),
    dict(name='p3-figure7', arxiv_id=V, column='full', page=5, anchor='Figure 7. Storing the KV cache of two requests at the same',
         figure=True, below=0, highlight=['Storing the KV cache of two requests']),
    dict(name='p3-one-block', arxiv_id=V, page=5, anchor='Again, vLLM dynamically assigns new physical blocks',
         highlight=['a new', 'physical block is only allocated when all previous blocks', 'limits all the memory wastes for a request',
                    'within one block', 'its KV blocks can be freed'],
         above=4, end='store the KV cache of other requests.'),
    dict(name='p3-figure8', arxiv_id=V, column='full', page=6, anchor='Figure 8. Parallel sampling example.', figure=True,
         below=0, highlight=['Parallel sampling example.']),
    dict(name='p3-cow', arxiv_id=V, page=6, anchor='Specifically, in Fig. 8, when sample',
         highlight=['reference count of the correspond-', 'is greater than 1', 'allocates a new physical block (physical block 3)',
                    'decreases the reference count to 1', 'directly writes'],
         above=4, end='cache to physical block 1.'),
    dict(name='p3-shared-prefix', arxiv_id=V, page=7, anchor='For this type of application, many user prompts share a',
         highlight=['store the KV cache', 'of the prefix in advance', 'how OS handles shared library across processes',
                    'only', 'needs to execute on the user’s task input'],
         above=4, end='needs to execute on the user’s task input.'),
    dict(name='p3-preempt', arxiv_id=V, page=7, anchor='LLM services face a unique challenge: the input prompts',
         highlight=['can run out of the GPU’s phys-', 'output lengths are not known a priori'],
         above=4, end='blocks to store the newly generated KV cache.'),
    dict(name='p3-swap-recompute', arxiv_id=V, page=7, anchor='Swapping. This is the classic technique used by most virtual',
         highlight=['we copy evicted blocks to', 'the CPU memory', 'recompute the KV', 'concatenated with the original user prompt as a new',
                    'one prompt phase iteration'],
         above=4, end='one prompt phase iteration.'),
    dict(name='p3-kernel-opt', arxiv_id=V, page=8, anchor='(2) Fusing block read and atten-',
         highlight=['read KV cache according to the block table', 'perform', 'attention operations on the fly',
                    'assign a GPU warp to read each block'],
         above=4, end='request batch.'),
    dict(name='p3-figure18', arxiv_id=V, column='full', page=11, anchor='Figure 18. Ablation experiments.', figure=True,
         below=0, highlight=['(a) Latency of attention kernels.']),
    dict(name='p3-kernel-overhead', arxiv_id=V, page=11, anchor='The dynamic block mapping in PagedAttention affects the',
         highlight=['extra over-', 'heads of accessing the block table', 'executing extra branches',
                    '20–26% higher attention kernel latency', 'only affects', 'the attention operator',
                    'significantly outperform FasterTransformer in', 'end-to-end performance'],
         above=4, end='end-to-end performance (§6).'),
    dict(name='p3-block-size', arxiv_id=V, page=11, anchor='In practice, we find that the block size 16 is large enough',
         highlight=['the block size 16 is large enough to', 'efficiently utilize the GPU', 'small enough to avoid signifi-',
                    'cant internal fragmentation'],
         above=4, end='vLLM sets its default block size as 16.'),
    dict(name='p3-swap-vs-recompute', arxiv_id=V, page=12, anchor='Our results reveal that swapping incurs excessive overhead',
         highlight=['swapping incurs excessive overhead', 'with small block sizes', 'never higher',
                    'than 20% of swapping’s latency', 'comparable end-to-end'],
         above=4, below=146),
    dict(name='p3-sarathi-chunk', arxiv_id=SARATHI, page=1, anchor='Sarathi-Serve is based',
         highlight=['chunked-prefills', 'stall-free', 'without pausing ongoing decodes'], above=4,
         end='without pausing ongoing decodes'),
]

# Figures that sit side by side in the paper: keep only one column of the full-width render (fractions of the width),
# then trim the white margin.
HCROP = {'p3-figure4': (0, 0.5), 'p3-figure5': (0.5, 1), 'p3-figure6': (0, 0.505), 'p3-figure7': (0.505, 1),
         'p3-figure8': (0, 0.5), 'p3-figure18': (0.5, 1)}


def hcrop(name):
    from PIL import Image, ImageChops
    path = os.path.join(paper_shots.OUT, name + '.png')
    im = Image.open(path).convert('RGB')
    a, b = HCROP[name]
    im = im.crop((int(a * im.width), 0, int(b * im.width), im.height))
    bbox = ImageChops.difference(im, Image.new('RGB', im.size, (255, 255, 255))).getbbox()
    if bbox:
        im = im.crop((max(0, bbox[0] - 12), max(0, bbox[1] - 12), min(im.width, bbox[2] + 12), min(im.height, bbox[3] + 12)))
    im.save(path)
    print(f'{name}: column crop {im.width}x{im.height}')


if __name__ == '__main__':
    only = set(filter(None, os.environ.get('ONLY', '').split(',')))
    for j in JOBS:
        if not only or j['name'] in only:
            try:
                print(shot(**j))
                if j['name'] in HCROP:
                    hcrop(j['name'])
            except (SystemExit, Exception) as e:
                print('FAILED', e)
