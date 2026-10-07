"""Highlighted excerpts for Chapter 4 (workflow patterns): Least-to-Most, Decomposed Prompting, AI Chains (chaining); RouteLLM,
FrugalGPT, Hybrid LLM (routing); More Agents, Universal Self-Consistency, Large Language Monkeys (parallel voting and sampling);
HuggingGPT, Magentic-One, AutoGen (orchestrator-workers); LLM-as-a-judge and G-Eval (evaluators). One figure or table and one
passage per paper. Usage: python shots_ch4.py  (ONLY=name1,name2 renders a subset)."""
import os
import paper_shots
from paper_shots import shot
from shots_ch3 import finish

# a non-arXiv PDF: OpenAI's "A practical guide to building agents", cached under a fixed name so paper_shots can open it like a paper
GUIDE_URL = 'https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf'
GUIDE = os.path.join(paper_shots.CACHE, 'openai-agents-guide.pdf')
if not os.path.exists(GUIDE):
    import urllib.request
    os.makedirs(paper_shots.CACHE, exist_ok=True)
    req = urllib.request.Request(GUIDE_URL, headers={'User-Agent': 'Mozilla/5.0'})
    open(GUIDE, 'wb').write(urllib.request.urlopen(req).read())

JOBS = [
    # ---- OpenAI, A practical guide to building agents (2025), page 16
    dict(name='ch4-openai-guide-multi', arxiv_id='openai-agents-guide', page=15, column='full', anchor='When to consider creating multiple agents',
         highlight=['maximize a single agent’s capabilities first', 'fail to follow complicated instructions'], above=4,
         end='improve performance.'),
    # ---- chaining
    dict(name='ch4-l2m-figure1', arxiv_id='2205.10625', page=1, figure=True, column='full',
         anchor='Figure 1: Least-to-most prompting solving a math word problem', highlight=[], below=0),
    dict(name='ch4-l2m-table11', arxiv_id='2205.10625', page=6, column='full', anchor='Overall, least-to-most prompting only slightly improves',
         highlight=['from 60.97%', 'from 39.07% to 45.23%'], above=4, end='The base language model is code-davinci-002.'),
    dict(name='ch4-decomp-figure1', arxiv_id='2210.02406', page=1, figure=True, column='full',
         anchor='Figure 1: While standard approaches only provide labeled examples', highlight=[], below=0),
    dict(name='ch4-decomp-abstract', arxiv_id='2210.02406', page=0, column='full', anchor='Few-shot prompting is a surprisingly powerful way',
         highlight=['shared library of prompting-based LLMs dedicated to these sub-tasks', 'trained models, or symbolic functions if desired'],
         above=4, end='symbolic functions if desired.'),
    dict(name='ch4-aichains-figure1', arxiv_id='2110.01691', page=1, figure=True, column='full',
         anchor='Figure 1: A walkthrough example illustrating the differences between no-Chaining', highlight=[], below=0),
    dict(name='ch4-aichains-abstract', arxiv_id='2110.01691', page=0, anchor='Although large language models (LLMs) have demonstrated',
         highlight=['the output of one step becomes the input for the next', 'Chaining not only improved the quality of task outcomes',
                    'debugged unexpected model outputs'], above=4, end='used in future applications.'),
    # ---- routing
    dict(name='ch4-routellm-figure1', arxiv_id='2406.18665', page=1, figure=True, column='full',
         anchor='Figure 1: Routing performance/cost trade-off between GPT-4 and Mixtral-8x7B', highlight=[], below=14),
    dict(name='ch4-routellm-table6', arxiv_id='2406.18665', page=8, column='full', anchor='CPT(50%)', above=8,
         highlight=['3.66 (95% GPT-4 quality)'], end='significantly reduce cost while maintaining response quality.'),
    dict(name='ch4-frugal-figure3', arxiv_id='2305.05176', page=6, figure=True, column='full',
         anchor='Figure 3: A case study of FrugalGPT on the HEADLINES dataset', highlight=['FrugalGPT reduces the cost by 80%'], below=0, fig_top=178),
    dict(name='ch4-frugal-table3', arxiv_id='2305.05176', page=7, column='full', anchor='Table 3: Cost savings by FrugalGPT to match',
         highlight=['98.3%', 'range from 50% to 98%'], above=4, end='utilized only for challenging queries detected by FrugalGPT.'),
    dict(name='ch4-hybrid-abstract', arxiv_id='2404.14618', page=0, column='full', anchor='Large language models (LLMs) excel in most NLP tasks',
         highlight=['router that assigns queries to the small or large model', 'up to 40% fewer calls to the large'], above=4, end='no drop in response quality.'),
    dict(name='ch4-hybrid-table1', arxiv_id='2404.14618', page=8, column='full', anchor='Table 1: Cost advantage v.s. performance drop',
         above=150, below=16, highlight=[]),
    # ---- parallelisation
    dict(name='ch4-moreagents-figure2', arxiv_id='2402.05120', page=3, figure=True, column='full',
         anchor='Figure 2: Illustration of Agent Forest', highlight=[], below=0),
    dict(name='ch4-moreagents-table2', arxiv_id='2402.05120', page=5, column='full', anchor='Table 2: Our method generally enhances performance',
         above=4, highlight=['0.59 ± 5e-4', '0.54 ± 3e-2'], end='which scores 54%.'),
    dict(name='ch4-usc-figure1', arxiv_id='2311.17311', page=1, figure=True, column='full',
         anchor='Figure 1: Overview of the Universal Self-Consistency workflow', highlight=[], below=0),
    dict(name='ch4-usc-tables', arxiv_id='2311.17311', page=4, column='full', anchor='Table 1: Accuracy on mathematical reasoning benchmarks',
         above=4, below=250, highlight=['SC-Exec (fuzzy match)']),
    dict(name='ch4-monkeys-abstract', arxiv_id='2407.21787', page=0, column='full', anchor='Scaling the amount of compute used to train language models',
         highlight=['from 15.9% with one sample to 56% with 250 samples', 'plateau beyond several hundred samples'], above=4, end='scale with the sample budget.'),
    dict(name='ch4-monkeys-figure7', arxiv_id='2407.21787', page=9, figure=True, column='full',
         anchor='Figure 7: Comparing coverage (performance with an oracle verifier)', highlight=['saturate before reaching 100 samples'], below=0),
    # ---- orchestrator-workers
    dict(name='ch4-hugginggpt-figure2', arxiv_id='2303.17580', page=2, figure=True, column='full',
         anchor='Figure 2: Overview of HuggingGPT', highlight=[], below=0),
    dict(name='ch4-hugginggpt-abstract', arxiv_id='2303.17580', page=0, column='full', anchor='Solving complicated AI tasks with different domains',
         highlight=['act as a controller to manage existing AI models', 'conduct task planning when receiving a user request'], above=4,
         end='towards the realization of artificial general intelligence.'),
    dict(name='ch4-magentic-figure2', arxiv_id='2411.04468', page=4, figure=True, column='full',
         anchor='Figure 2: Magentic-One features an Orchestrator agent that implements two loops', highlight=[], below=0),
    dict(name='ch4-magentic-ledgers', arxiv_id='2411.04468', page=5, column='full', anchor='At a high level, the workflow',
         highlight=['the outer loop maintains the task ledger', 'we force all agents to clear their contexts and reset their states'],
         above=4, end='Once the plan is formed, the inner loop is initiated.'),
    dict(name='ch4-magentic-table1', arxiv_id='2411.04468', page=11, column='full', anchor='omne v0.1 (GPT-4o, o1)', above=40, below=320,
         highlight=['38.00±5.5', '92.00±3.1']),
    dict(name='ch4-autogen-tables', arxiv_id='2308.08155', page=27, column='full', anchor='Table 5: Number of successes on the 12 tasks',
         above=4, highlight=[], below=135),
    # ---- evaluators
    dict(name='ch4-judge-abstract', arxiv_id='2306.05685', page=0, column='full', anchor='Evaluating large language model (LLM) based chat assistants',
         highlight=['position, verbosity, and self-enhancement biases', 'achieving over 80% agreement'], above=4, end='which are otherwise very expensive to obtain.'),
    dict(name='ch4-judge-table2', arxiv_id='2306.05685', page=4, column='full', anchor='Table 2: Position bias of different LLM judges',
         above=4, below=130, highlight=['75.0%']),
    dict(name='ch4-geval-figure1', arxiv_id='2303.16634', page=1, figure=True, column='full',
         anchor='Figure 1: The overall framework of G-EVAL', highlight=[], below=0),
    dict(name='ch4-geval-table1', arxiv_id='2303.16634', page=3, column='full', anchor='Table 1: Summary-level Spearman', above=250, below=40,
         highlight=['G-EVAL-4']),
]

TRIM = {'ch4-decomp-abstract': 130, 'ch4-hybrid-abstract': 130, 'ch4-monkeys-abstract': 130, 'ch4-hugginggpt-abstract': 130,
        'ch4-judge-abstract': 130}

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    done, failed = [], []
    for job in JOBS:
        if only and job['name'] not in only.split(','):
            continue
        try:
            shot(**job)
            done.append(job['name'])
        except (SystemExit, Exception) as e:
            failed.append((job['name'], str(e)[:160]))
    finish(done, TRIM)
    for n, e in failed:
        print('FAILED', n, e)
