"""Highlighted excerpts of the Meta-Harness paper for Part 6 (build it from scratch): Figure 2, §1 (the coding-agent
proposer), §3 (filesystem, Pareto, search vs test, practical implementation, footnote 1), Algorithm 1,
§4.1 (online setup), Table 8, Appendix B.1 (TF-IDF, draft verification), Figure 7, Appendix C.1 (Banking77),
Appendix D (skill, search set, logs, validation)."""
from paper_shots import run

JOBS = [
    # ---------------------------------------------------------------- page 2 (index 1): Figure 2, §1
    dict(name='p6-figure2', page=1, anchor='Figure 2: Meta-Harness search loop.',
         highlight=['reads a filesystem containing all prior candidates', 'stored in the filesystem in a new directory'],
         figure=True, column='full', below=0),
    dict(name='p6-coding-agent', page=1, anchor='We address this limitation with Meta-Harness',
         highlight=['the amount of experience quickly exceeds context limits', 'decide what to inspect',
                    'via standard operations such as grep and cat'],
         above=4, end='via standard operations such as grep and cat rather than ingesting them as a single prompt.'),
    # ---------------------------------------------------------------- page 4 (index 3): §3
    dict(name='p6-pareto', page=3, anchor='When multiple objectives are relevant',
         highlight=['Pareto dominance', 'report the resulting frontier'], above=4, end='report the resulting frontier.'),
    dict(name='p6-directory', page=3, anchor='Each evaluated harness',
         highlight=['contributes a directory containing its source code, scores, and execution traces',
                    'grep and cat'],
         above=4, end='before generating a new harness.'),
    dict(name='p6-never-sees', page=3, anchor='The proposer never sees test-set results',
         highlight=['The proposer never sees test-set results', 'search set'], above=4, end='runs.'),
    dict(name='p6-footnote', page=3, anchor='we think this workflow only became practical recently',
         highlight=['only became practical recently'], above=4, below=12),
    # ---------------------------------------------------------------- page 5 (index 4): Algorithm 1, practical, §4.1
    dict(name='p6-alg1', page=4, anchor='Algorithm 1 Meta-Harness outer loop over harnesses',
         highlight=['passes interface validation', 'return Pareto frontier of harnesses stored in D'],
         above=4, end='return Pareto frontier of harnesses stored in D'),
    dict(name='p6-practical', page=4, anchor='Practical implementation.',
         highlight=['single-file Python program', 'minimal domain-specific skill', 'always frozen',
                    'roughly 60 harnesses over 20 iterations'],
         above=4, end='roughly 60 harnesses over 20 iterations.'),
    dict(name='p6-online', page=4, anchor='We follow the online text classification setup',
         highlight=['receives labeled examples one at a time, updates its memory, and is evaluated on a held',
                    'out test set'],
         above=4, below=24),
    # ---------------------------------------------------------------- page 15 (index 14): Table 8
    dict(name='p6-table8', page=14, anchor='Table 8 summarizes the results.',
         highlight=['median of 82 files per iteration', 'non-Markovian', 'Files read per iteration (median)'],
         above=4, column='full', end='attention to prior source code and execution traces.'),
    # ---------------------------------------------------------------- page 18, 19 (index 17, 18): Appendix B.1
    dict(name='p6-draft-verify', page=17, anchor='Meta-Harness (Draft Verification). The corresponding discovered file',
         highlight=['5 confirmers with the same label and 5 challengers with different labels',
                    'falls back to a standard single-call few-shot prompt'],
         above=4, end='single-call few-shot prompt.'),
    dict(name='p6-tfidf', page=18, anchor='Meta-Harness (Label-Primed Query). The corresponding',
         highlight=['one query-relevant example per label', 'query-anchored contrastive pairs',
                    'TF-IDF retrieval over past labeled examples'], above=4, end='from the same local neighborhood.'),
    # ---------------------------------------------------------------- page 20 (index 19): Figure 7
    dict(name='p6-figure7', page=19, anchor='Figure 7: Search-set vs. test accuracy',
         highlight=['The dashed diagonal is y=x'], figure=True, column='full', below=0),
    # ---------------------------------------------------------------- page 23 (index 22): Appendix C.1 Banking77
    dict(name='p6-banking77', page=22, anchor='Banking77 is a fine-grained intent classification',
         highlight=['labeled with 77 intents'], above=4, end='large label space.'),
    # ---------------------------------------------------------------- page 24, 25 (index 23, 24): Appendix D
    dict(name='p6-skill', page=23, anchor='Write a good skill.',
         highlight=['primary interface for steering the search', 'constrain outputs and safety-relevant behavior',
                    'not the proposer’s diagnosis procedure'],
         above=4, end='before committing to a full run.'),
    dict(name='p6-search-set', page=23, anchor='Start with a baseline harness and a search set that is hard for it',
         highlight=['The search has little to optimize if the baseline already saturates the evaluation',
                    'roughly 50 full evaluations per run'],
         above=4, below=60),
    dict(name='p6-logs', page=24, anchor='Log everything in a format that is easy to navigate',
         highlight=['machine-readable formats such as JSON', 'lists the Pareto frontier, shows top-k harnesses, and diffs'],
         above=4, end='wasted on navigation.'),
    dict(name='p6-validation', page=24, anchor='Lightweight validation before expensive benchmarks',
         highlight=['imports the module, instantiates the class, and calls both methods',
                    'Automate evaluation outside the proposer'],
         above=4, end='write results to the filesystem.'),
]

if __name__ == '__main__':
    run(JOBS)
