"""Highlighted excerpts for the Background page (what a harness is, and the optimizers before Meta-Harness).
Meta-Harness (2603.28052) plus the papers it builds on or compares against: ReAct (2210.03629), OPRO (2309.03409),
DSPy (2310.03714), TextGrad (2406.07496), GEPA (2507.19457), AlphaEvolve (2506.13131), ACE (2510.04618),
TTT-Discover (2601.16175). Run: python shots_partb.py   (or ONLY=pb-name1,pb-name2)"""
from paper_shots import run, SAM as MH

REACT, OPRO, DSPY, TGRAD, GEPA, AEVO, ACE, TTTD = ('2210.03629', '2309.03409', '2310.03714', '2406.07496',
                                                   '2507.19457', '2506.13131', '2510.04618', '2601.16175')

JOBS = [
    # ---------------------------------------------------------------- what a harness is (Meta-Harness itself)
    dict(name='pb-harness-def', page=3, anchor='Objective. A harness is a stateful program',
         highlight=['A harness is a stateful program that wraps a language model and determines',
                    'what context the model sees at each step',
                    'constructs prompts for M, the', 'model responds, and the harness updates its state after each interaction'],
         above=4, end='updates its state after each interaction.'),
    dict(name='pb-memory', page=2, anchor='External memory and adaptive access',
         highlight=['accesses', 'adaptively, rather than consuming them in a single pass',
                    'retrieval-augmented', 'generation', 'memory-based agents'],
         hl_nth={'accesses': 0}, above=4, end='context-management procedures themselves.'),
    # ---------------------------------------------------------------- ReAct
    dict(name='pb-react-abstract', arxiv_id=REACT, page=0, anchor='abilities for reasoning (e.g. chain-of-thought prompting)',
         highlight=['generate both reasoning traces and task-speciﬁc actions', 'in an interleaved manner',
                    'gather additional information', 'from external sources such as knowledge bases or environments'],
         above=4, end='from external sources such as knowledge bases or environments.'),
    dict(name='pb-react-setup', arxiv_id=REACT, page=2, anchor='Consider a general setup of an agent interacting',
         highlight=['following some policy π(at|ct), where ct = (o1, a1, · · · , ot−1, at−1, ot) is the context to the agent',
                    'we augment the agent’s action space to', 'A ∪L, where L is the', 'space of language'],
         above=4, end='to support future reasoning or acting.'),
    # ---------------------------------------------------------------- frameworks: DSPy
    dict(name='pb-frameworks', page=25, anchor='Prompt orchestration frameworks.',
         highlight=['still typically require manual design of retrieval', 'policies, memory updates, and orchestration logic',
                    'searches over the implementation of these policies in executable code'],
         above=4, end='treating the harness itself as the optimization target.'),
    dict(name='pb-dspy-abstract', arxiv_id=DSPY, page=0, anchor='The ML community is rapidly exploring techniques',
         highlight=['hard-coded', '“prompt templates”, i.e. lengthy strings discovered via trial and error',
                    'compiler', 'that will optimize any DSPy pipeline to maximize a given metric'],
         hl_nth={'compiler': 0}, above=4, end='that will optimize any DSPy pipeline to maximize a given metric.'),
    # ---------------------------------------------------------------- OPRO
    dict(name='pb-opro-abstract', arxiv_id=OPRO, page=0, anchor='Optimization is ubiquitous',
         highlight=['the LLM generates new solutions from the prompt that',
                    'contains previously generated solutions with their values',
                    'evaluated and added to the prompt for the next optimization step'],
         above=4, end='added to the prompt for the next optimization step.'),
    dict(name='pb-opro-trajectory', arxiv_id=OPRO, page=3, anchor='Optimization trajectory. Besides understanding',
         highlight=['past solutions and their optimization scores, sorted in the ascending', 'order'],
         hl_nth={'order': 0}, above=4, end='explicitly deﬁning how the solution should be updated.'),
    # ---------------------------------------------------------------- TextGrad
    dict(name='pb-textgrad-tgd', arxiv_id=TGRAD, page=4, anchor='In standard gradient descent, the current value',
         highlight=['Textual Gradient Descent (TGD)', 'Incorporate the criticisms, and produce a new variable'],
         above=4, end='Incorporate the criticisms, and produce a new variable.).', column='full'),
    # ---------------------------------------------------------------- GEPA
    dict(name='pb-gepa-abstract', arxiv_id=GEPA, page=0, anchor='We argue that the interpretable',
         highlight=['samples trajectories (e.g., reasoning, tool calls, and tool', 'outputs)',
                    'reﬂects on them in natural language to diagnose problems',
                    'combine complementary lessons from the Pareto frontier'],
         above=4, end='attempts. As a result of'),
    dict(name='pb-gepa-mh', page=24, anchor='GEPA [1] is the closest text optimizer',
         highlight=['closest text optimizer in terms of feedback richness',
                    'GEPA operates on one candidate at a time', 'lets the agent decide what to examine'],
         above=4, end='lets the agent decide what to examine.'),
    # ---------------------------------------------------------------- AlphaEvolve / OpenEvolve
    dict(name='pb-alphaevolve-abstract', arxiv_id=AEVO, page=0, anchor='In this white paper, we present AlphaEvolve',
         highlight=['improve an algorithm by making direct changes to the code',
                    'evolutionary approach, continuously receiving feedback from one or more evaluators'],
         above=4, end='potentially leading to new scientiﬁc and practical discoveries.'),
    dict(name='pb-alphaevolve-mh', page=24, anchor='AlphaEvolve [35] and OpenEvolve [43] evolve code',
         highlight=['program database', 'with scalar scores', 'tournament-selected parents',
                    'single stateless function with a clean scalar objective',
                    'harnesses are stateful programs that'],
         hl_nth={'program database': 0}, above=4, end='through an entire evaluation sequence.'),
    # ---------------------------------------------------------------- ACE
    dict(name='pb-ace-abstract', arxiv_id=ACE, page=0, anchor='Large language model (LLM) applications such as agents',
         highlight=['context adaptation', 'brevity bias', 'context collapse',
                    'treats contexts as evolving playbooks', 'generation, reﬂection, and curation'],
         hl_nth={'context adaptation': 0, 'context collapse': 0}, above=4, end='and scale with long-context models.'),
    dict(name='pb-ace-mh', page=6, anchor='Comparison vs state-of-the-art harnesses',
         highlight=['reﬂective', 'memory curation to build context over time',
                    'maintains and evolves a library of natural-language skills'],
         above=4, end='for context construction.'),
    # ---------------------------------------------------------------- TTT-Discover
    dict(name='pb-ttt-abstract', arxiv_id=TTTD, page=0, anchor='How can we use AI to discover a new state',
         highlight=['performs search by prompting a frozen LLM', 'We perform',
                    'reinforcement learning at test time, so the LLM can continue to train'],
         above=4, end='rather than generalize to other problems.'),
    dict(name='pb-ttt-puct', arxiv_id=TTTD, page=5, anchor='PUCT. We select initial states using',
         highlight=['Q(s) + c · P (s) ·', '1 + T /(1 + n(s))'],
         above=4, end='T is the total number of expansions, and c is the exploration coefﬁcient.'),
    dict(name='pb-compared', page=5, anchor='Best-of-N: independent samples from the seed',
         highlight=['evolutionary search over programs with LLM mutation',
                    'we use only the text-optimization component of their method',
                    'proposal selection via the PUCT reuse rule'],
         above=4, end='proposal selection via the PUCT reuse rule.'),
    # ---------------------------------------------------------------- Table 1: what each optimizer sees
    dict(name='pb-table1', page=1, anchor='Table 1: Comparison of text optimization methods',
         highlight=['past (solution, score) pairs', 'all logs and scores'], figure=True, fig_top=118, below=0),
]

# Pixels to cut after rendering: (left, top, right, bottom). Abstract crops carry arXiv's rotated side stamp on the
# left; pb-ace-mh sits in a column next to Table 4, which the automatic column finder does not see.
TRIM = {'pb-react-abstract': (310, 0, 0, 0), 'pb-dspy-abstract': (310, 0, 0, 0), 'pb-opro-abstract': (310, 0, 0, 0),
        'pb-gepa-abstract': (270, 0, 0, 0), 'pb-alphaevolve-abstract': (100, 0, 0, 0), 'pb-ace-abstract': (310, 0, 0, 0),
        'pb-ttt-abstract': (235, 0, 0, 0), 'pb-ace-mh': (0, 0, 442, 8), 'pb-table1': (0, 30, 0, 0)}

if __name__ == '__main__':
    import os
    from PIL import Image
    from paper_shots import OUT
    run(JOBS)
    only = os.environ.get('ONLY')
    for name, (l, t, r, b) in TRIM.items():
        if only and name not in only.split(','):
            continue
        path = os.path.join(OUT, name + '.png')
        im = Image.open(path)
        im.crop((l, t, im.width - r, im.height - b)).save(path)
        print(f'{name}: trimmed to {im.width - l - r}x{im.height - t - b}')
