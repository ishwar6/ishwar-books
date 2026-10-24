"""Highlighted excerpts of the paper for Part 6 (build it from scratch): §4 to §4.5 and §5, the passages that
each piece of the minivllm package implements.
Run: python shots_part6.py   (or ONLY=p6-eq4,p6-figure6 python shots_part6.py)"""
from paper_shots import run

JOBS = [
    dict(name='p6-figure4', page=4, anchor='Figure 4. vLLM system overview.', figure=True, column='full', below=0,
         highlight=['vLLM system overview']),
    dict(name='p6-method', page=4, anchor='In this work, we develop a new attention algorithm, Page',
         highlight=['centralized scheduler to', 'KV cache manager effectively manages the KV cache in a', 'paged fashion'],
         above=4, end='the instructions sent by the centralized scheduler'),
    dict(name='p6-blocks', page=4, anchor='To address the memory challenges in',
         highlight=['non-contiguous memory space', 'partitions the KV cache of each', 'fixed number of'],
         above=4, end='which we denote as KV'),
    dict(name='p6-eq4', page=4, anchor='block size (B). Denote the key block',
         highlight=['Denote the key block', 'wise computation', 'row vector of atten'],
         above=4, end='tion score on'),
    dict(name='p6-kernel', page=4, anchor='During the attention computation, the PagedAttention',
         highlight=['identifies and fetches different KV blocks separately', 'not contiguous on the physical memory'],
         above=4, end='to derive the final attention output'),
    dict(name='p6-vm', page=4, anchor='The key idea behind',
         highlight=['logical pages to physical pages', 'non-contiguous physical memory pages',
                    'fixed-size KV blocks, like pages in virtual memory'],
         above=4, end='cache as fixed-size KV blocks, like pages in virtual memory'),
    dict(name='p6-logical', page=4, anchor='is represented as a series of logical',
         highlight=['filled from left to right', 'unfilled positions', 'reserved for future generations',
                    'allocates a contiguous chunk of GPU DRAM'],
         above=4, end='engine allocates a contiguous chunk of GPU DRAM and'),
    dict(name='p6-table', page=5, anchor='divides it into physical KV blocks',
         highlight=['maintains block tables', 'the number of filled positions', 'without reserving it for all positions in'],
         above=4, end='eliminates most memory waste in existing systems'),
    dict(name='p6-figure6', page=5, anchor='Figure 6. Block table translation in vLLM.', figure=True, column='full', below=0,
         highlight=['Block table translation', 'Storing the KV cache of two requests']),
    dict(name='p6-step1', page=5, anchor='Next, we walk through an example, as in Fig. 6',
         highlight=['does not require', 'reserving the memory for the maximum possible generated', 'necessary KV blocks',
                    'The remaining slot is reserved for the'],
         above=4, end='subsequent autoregressive generation phase'),
    dict(name='p6-step23', page=5, anchor='first autoregressive decoding step',
         highlight=['one slot remains available in the last logical block', 'allocates a', 'new physical block'],
         above=4, end='stores this mapping in the block table'),
    dict(name='p6-iteration', page=5, anchor='Globally, for each decoding iteration',
         highlight=['selects a set of candidate sequences for batching', 'concatenates all the input tokens'],
         above=4, end='of the current iteration (i.e., all tokens for prompt phase'),
    dict(name='p6-iteration-2', page=5, anchor='requests and the latest tokens for generation phase requests',
         highlight=['as one sequence and feeds it into the LLM', 'PagedAttention kernel to access'],
         above=4, end='and saves the newly generated KV cache into the physical KV blocks'),
    dict(name='p6-one-block', page=5, anchor='Again, vLLM dynamically assigns new physical blocks',
         highlight=['a new', 'physical block is only allocated when all previous blocks', 'limits all the memory wastes for a request',
                    'within one block', 'its KV blocks can be freed'],
         above=4, end='can be freed to store the KV cache of other requests'),
    dict(name='p6-refcount', page=6, anchor='Fig. 8 shows an example of parallel decoding for two out',
         highlight=['reserve space for one copy of the', 'reference count for', 'each physical block'],
         above=4, end='physical blocks 7 and 1 are both 2'),
    dict(name='p6-cow', page=6, anchor='At the generation phase, the two outputs sample different',
         highlight=['copy-on-write mechanism at the block granularity', 'reference count of the correspond',
                    'decreases the reference count to 1', 'directly writes'],
         above=4, end='A2 directly writes its newly generated KV cache to physical block 1'),
    dict(name='p6-figure8', page=6, anchor='Figure 8. Parallel sampling example.', figure=True, column='full', below=0,
         highlight=['Parallel sampling example']),
    dict(name='p6-mixed', page=7, anchor='Mixed decoding methods.',
         highlight=['common mapping layer', 'only see a list of physical block IDs'],
         above=4, end='need to handle sharing patterns across sequences'),
    dict(name='p6-fcfs', page=7, anchor='When the request traffic surpasses the',
         highlight=['first-come-first-serve (FCFS) scheduling policy', 'arrived requests are served first and the latest requests are',
                    'preempted first'],
         above=4, end='requests are served first and the latest requests are preempted first'),
    dict(name='p6-evict', page=7, anchor='Which blocks should it evict',
         highlight=['Which blocks should it evict', 'How to recover', 'all-or-nothing eviction policy',
                    'gang-scheduled as a sequence'],
         above=16, end='we consider two techniques'),
    dict(name='p6-recompute', page=7, anchor='Recomputation. In this case, we simply recompute',
         highlight=['recompute the KV', 'cache when the preempted sequences are rescheduled',
                    'concatenated with the original user prompt as a new', 'one prompt phase iteration'],
         above=4, end='generated in one prompt phase iteration'),
    dict(name='p6-impl', page=8, anchor='vLLM is an end-to-end serving system with a FastAPI',
         highlight=['8.5K lines of Python and 2K lines of', 'scheduler and the block manager in Python',
                    'custom CUDA kernels for key operations'],
         above=16, end='developing custom CUDA kernels for key operations such as PagedAttention'),
    dict(name='p6-kernels', page=8, anchor='Since PagedAttention introduces memory access patterns',
         highlight=['Fused reshape and block write', 'Fusing block read and attention', 'Fused block copy',
                    'batches the copy operations for different blocks into a single'],
         above=16, end='into a single kernel launch'),
    dict(name='p6-fork', page=8, anchor='vLLM implements various decoding algorithms using three',
         highlight=['fork, append, and free', 'creates a new sequence from an existing one',
                    'appends a new token to the sequence', 'deletes the sequence'],
         above=16, end='ing algorithms can also be supported by combining these methods'),
]



def keep_left(name, share):
    """Figures 4 and 8 share their row with another figure: keep only the left part of the full-width crop."""
    import os
    from PIL import Image
    from paper_shots import OUT
    path = os.path.join(OUT, name + '.png')
    im = Image.open(path)
    im.crop((0, 0, int(im.width * share), im.height)).save(path, optimize=True)


if __name__ == '__main__':
    run(JOBS)
    import os
    if not os.environ.get('ONLY') or 'p6-figure8' in os.environ['ONLY']:
        keep_left('p6-figure8', 0.5)
    if not os.environ.get('ONLY') or 'p6-figure4' in os.environ['ONLY']:
        keep_left('p6-figure4', 0.46)              # Figure 5 sits to the right of Figure 4 on the same row
