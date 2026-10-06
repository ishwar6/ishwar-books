"""Highlighted excerpts for Chapter 3 (reasoning patterns): chain of thought, self-consistency, zero-shot CoT, the two counter-evidence
papers (unfaithful explanations, cannot self-correct yet), ReAct, Plan-and-Solve, ReWOO, LLMCompiler, Reflexion, Self-Refine, CRITIC,
Tree of Thoughts, LATS, CodeAct, SWE-agent, Agentless, DeepSeek-R1 (version 1) and the o1 system card. One figure or table and one
passage per paper. Usage: python shots_ch3.py  (ONLY=name1,name2 renders a subset)."""
import os
import paper_shots
from paper_shots import shot

JOBS = [
    # ---- chain of thought (Wei et al. 2022)
    dict(name='ch3-cot-figure1', arxiv_id='2201.11903', page=0, figure=True, column='full',
         anchor='Figure 1: Chain-of-thought prompting enables large language models', highlight=['Chain-of-thought reasoning processes are highlighted'], below=0),
    dict(name='ch3-cot-abstract', arxiv_id='2201.11903', page=0, column='full',
         anchor='We explore how generating a chain of thought',
         highlight=['a series of intermediate reasoning', 'emerge naturally in sufficiently large language models', 'surpassing even finetuned GPT-3 with a verifier'],
         above=4, end='finetuned GPT-3 with a verifier.'),
    # ---- self-consistency (Wang et al. 2022)
    dict(name='ch3-sc-abstract', arxiv_id='2203.11171', page=0, column='full',
         anchor='Chain-of-thought prompting combined with pre-trained large language models',
         highlight=['samples a diverse set of reasoning paths', 'selects the most consistent answer', 'GSM8K (+17.9%)'],
         above=4, end='ARC-challenge (+3.9%).'),
    dict(name='ch3-sc-table2', arxiv_id='2203.11171', page=4, column='full', anchor='Previous SoTA', above=16,
         highlight=['74.4 (+17.9)', '78.0 (+17.9)'], end='The best performance for each task is shown in bold.'),
    # ---- zero-shot CoT (Kojima et al. 2022)
    dict(name='ch3-zeroshot-figure1', arxiv_id='2205.11916', page=1, figure=True, column='full',
         anchor='Figure 1: Example inputs and outputs of GPT-3 with (a) standard Few-shot', highlight=['Zero-shot-CoT'], below=0),
    dict(name='ch3-zeroshot-abstract', arxiv_id='2205.11916', page=0, column='full',
         anchor='While these successes are often attributed to',
         highlight=['decent zero-shot reasoners', 'MultiArith from 17.7% to 78.7%', 'GSM8K from 10.4% to 40.7%'],
         above=4, end='540B parameter PaLM.'),
    # ---- unfaithful explanations (Turpin et al. 2023)
    dict(name='ch3-unfaithful-abstract', arxiv_id='2305.04388', page=0, column='full',
         anchor='However, we find that CoT explanations can systematically misrepresent',
         highlight=['systematically misrepresent the true reason', 'drop by as much as 36%', 'plausible yet misleading'],
         above=4, end='in favor of alternative methods.'),
    dict(name='ch3-unfaithful-figure1', arxiv_id='2305.04388', page=4, figure=True, column='full',
         anchor='Figure 1: Accuracy micro-averaged across BBH tasks', highlight=['CoT explanations do not mention the biasing feature'], below=0),
    # ---- cannot self-correct yet (Huang et al. 2023)
    dict(name='ch3-selfcorrect-abstract', arxiv_id='2310.01798', page=0, column='full',
         anchor='Central to our investigation is the notion',
         highlight=['intrinsic self-correction', 'struggle to self-correct their responses without external feedback', 'performance even degrades after self-correction'],
         above=4, end='applications in this field.'),
    dict(name='ch3-selfcorrect-table3', arxiv_id='2310.01798', page=3, column='full',
         anchor='Table 3: Results of GPT-3.5 and GPT-4 on reasoning benchmarks with intrinsic self-correction', above=4, below=118),
    dict(name='ch3-selfcorrect-why', arxiv_id='2310.01798', page=3, column='full',
         anchor='For GSM8K, 74.7% of the time',
         highlight=['more likely to modify a correct answer to an incorrect one', 'LLMs cannot properly judge the correctness of their reasoning'],
         above=4, end='correctness of their reasoning.'),
    # ---- ReAct (Yao et al. 2022)
    dict(name='ch3-react-table1', arxiv_id='2210.03629', page=4, anchor='Table 1: PaLM-540B prompting results on', above=150,
         highlight=['CoT-SC →ReAct', 'ReAct→CoT-SC'], end='number of CoT-SC samples used.'),
    dict(name='ch3-react-table2', arxiv_id='2210.03629', page=5, column='full', anchor='Table 2: Types of success and failure modes of ReAct and CoT on HotpotQA',
         above=140, below=16, highlight=['failing to recover from repetitive steps', 'Hallucination']),
    dict(name='ch3-react-errors', arxiv_id='2210.03629', page=5, column='full',
         anchor='While interleaving reasoning, action and observation steps',
         highlight=['repetitively generates the previous thoughts and actions', 'Non-informative search, which counts for 23% of the error cases'],
         above=4, end='combining two methods.'),
    dict(name='ch3-react-table3', arxiv_id='2210.03629', page=7, column='full', anchor='Table 3: AlfWorld task-specific success rates', above=130, below=30,
         highlight=['ReAct (best of 6)']),
    # ---- Plan-and-Solve (Wang et al. 2023)
    dict(name='ch3-ps-abstract', arxiv_id='2305.04091', page=0,
         anchor='Despite the success of Zero-shot-CoT, it still suffers from three pitfalls',
         highlight=['calculation errors, missing-step errors, and semantic misunderstanding errors', 'devising a plan to divide the entire task into smaller subtasks'],
         above=4, end='according to the plan.'),
    dict(name='ch3-ps-prompt', arxiv_id='2305.04091', page=2,
         anchor='Thus, the prompt would be',
         highlight=['devise a plan to solve the problem', 'carry out the plan and solve the problem step by step'],
         above=4, end='generating output by default.'),
    dict(name='ch3-ps-table2', arxiv_id='2305.04091', page=5, column='full', anchor='Table 2: Accuracy comparison on six math reasoning datasets', above=4, below=108,
         highlight=['PS+ (ours)']),
    # ---- ReWOO (Xu et al. 2023)
    dict(name='ch3-rewoo-abstract', arxiv_id='2305.18323', page=0, column='full',
         anchor='Such a paradigm, though straightforward and easy to implement',
         highlight=['redundant prompts and repeated execution', 'detaches the reasoning process from external observations', '5× token efficiency and 4% accuracy improvement'],
         above=4, end='under tool-failure scenarios.'),
    dict(name='ch3-rewoo-figure1', arxiv_id='2305.18323', page=1, figure=True, column='full', anchor='Figure 1: Workflow of ReWOO', highlight=['Planner', 'Solver'], below=0),
    dict(name='ch3-rewoo-table2', arxiv_id='2305.18323', page=6, column='full', anchor='Paradigm', above=8, below=150,
         highlight=['9795.1', '1986.2']),
    # ---- LLMCompiler (Kim et al. 2023)
    dict(name='ch3-llmcompiler-abstract', arxiv_id='2312.04511', page=0,
         anchor='However, current methods for function calling often require sequential',
         highlight=['sequential reasoning and acting for each function', 'latency speedup of up to 3.7×, cost savings of up to 6.7×'],
         above=4, end='compared to ReAct.'),
    dict(name='ch3-llmcompiler-table1', arxiv_id='2312.04511', page=5, column='full', anchor='Table 1. Accuracy and latency comparison of LLMCompiler', above=4, below=215,
         highlight=['looping and early stopping']),
    # ---- Reflexion (Shinn et al. 2023)
    dict(name='ch3-reflexion-figure2', arxiv_id='2303.11366', page=3, figure=True, column='full', anchor='Figure 2: (a) Diagram of Reflexion.', highlight=['Reflexion reinforcement algorithm'], below=0),
    dict(name='ch3-reflexion-tests', arxiv_id='2303.11366', page=6, column='full',
         anchor='The task of programming presents a unique opportunity',
         highlight=['self-generated unit test suites'], above=4, end='max memory limit of 1 experience.'),
    dict(name='ch3-reflexion-table1', arxiv_id='2303.11366', page=6, column='full', anchor='Table 1: Pass@1 accuracy for various model-strategy-language combinations',
         above=105, below=30, highlight=['91.0']),
    # ---- Self-Refine (Madaan et al. 2023)
    dict(name='ch3-selfrefine-table1', arxiv_id='2303.17651', page=4, column='full', anchor='Table 1: SELF-REFINE results on various tasks', above=135, below=30,
         highlight=['Math Reasoning']),
    dict(name='ch3-selfrefine-math', arxiv_id='2303.17651', page=4, column='full',
         anchor='The modest performance gains in Math Reasoning can be traced back',
         highlight=['inability to accurately identify whether there is any error'], above=4, below=30),
    # ---- CRITIC (Gou et al. 2023)
    dict(name='ch3-critic-abstract', arxiv_id='2305.11738', page=0, column='full',
         anchor='Unlike these models, humans typically utilize external tools',
         highlight=['using a search engine for fact-checking, or a code interpreter for debugging', 'crucial importance of external feedback'],
         above=4, end='external feedback in promoting the ongoing'),
    dict(name='ch3-critic-figure1', arxiv_id='2305.11738', page=1, figure=True, column='full', anchor='Figure 1: The CRITIC framework consists of two steps', highlight=['verify-then-correct'], below=0),
    dict(name='ch3-critic-table2', arxiv_id='2305.11738', page=6, anchor='Table 2: Mathematical program synthesis results.', above=4, below=235, highlight=['w/o Tool']),
    # ---- Tree of Thoughts (Yao et al. 2023)
    dict(name='ch3-tot-figure1', arxiv_id='2305.10601', page=1, figure=True, column='full', anchor='Figure 1: Schematic illustrating various approaches', highlight=['Tree of Thoughts'], below=0),
    dict(name='ch3-tot-table2', arxiv_id='2305.10601', page=5, anchor='Table 2: Game of 24 Results.', above=128, below=10, highlight=['74%', '4.0%']),
    dict(name='ch3-tot-cost', arxiv_id='2305.10601', page=13, column='full',
         anchor='Running ToT requires significantly more computations than IO or CoT prompting',
         highlight=['5.5k completion tokens, close to 100 CoT trials'], above=4, end='Table 7: Cost analysis on Game of 24.'),
    # ---- LATS (Zhou et al. 2023)
    dict(name='ch3-lats-figure1', arxiv_id='2310.04406', page=0, figure=True, anchor='Figure 1. Overview of LATS.', highlight=['MCTS-based'], below=22),
    dict(name='ch3-lats-table3', arxiv_id='2310.04406', page=5, anchor='Table 3. GPT-3.5 acting-based prompting results on HotpotQA.', above=150, below=4,
         highlight=['LATS (CoT + ReAct)'], end='adaptations of search algorithms to decision'),
    dict(name='ch3-lats-cost', arxiv_id='2310.04406', page=8, anchor='Table 10. Comparison of the cost of different methods', above=168, below=4,
         highlight=['# of Nodes'], end='at various k'),
    # ---- CodeAct (Wang et al. 2024)
    dict(name='ch3-codeact-abstract', arxiv_id='2402.01030', page=0,
         anchor='This work proposes to use executable Python code to consolidate',
         highlight=['executable Python code to consolidate', 'up to 20% higher success rate'], above=4, end='20% higher success rate).'),
    dict(name='ch3-codeact-figure1', arxiv_id='2402.01030', page=1, figure=True, column='full', anchor='Figure 1: Comparison between CodeAct and Text / JSON as action.', highlight=['CodeAct'], below=0),
    dict(name='ch3-codeact-table3', arxiv_id='2402.01030', page=4, anchor='Table 3: Success rates (higher the better) and average turns required', above=4, below=225,
         highlight=['Avg. Turns']),
    # ---- SWE-agent (Yang et al. 2024)
    dict(name='ch3-sweagent-figure1', arxiv_id='2405.15793', page=0, figure=True, column='full', anchor='Figure 1: SWE-agent is an LM interacting with a computer', highlight=['agent-computer interface'], below=0),
    dict(name='ch3-sweagent-table1', arxiv_id='2405.15793', page=5, column='full', anchor='Table 1: Main results for SWE-agent performance', above=4, below=183,
         highlight=['12.47', '18.00']),
    dict(name='ch3-sweagent-table3', arxiv_id='2405.15793', page=5, column='full', anchor='Table 3: SWE-bench Lite performance under ablations', above=4, below=120,
         highlight=['w/ linting', 'Last 5 Obs.']),
    dict(name='ch3-sweagent-edits', arxiv_id='2405.15793', page=7, column='full',
         anchor='Editing remains challenging for agents.',
         highlight=['1,185 (51.7%)', 'Agents succeed quickly and fail slowly'], above=4, end='before exhausting their cost'),
    dict(name='ch3-sweagent-figure8', arxiv_id='2405.15793', page=7, figure=True, anchor='Figure 8: Failure mode distribution', highlight=['Failure mode distribution'], below=0),
    # ---- Agentless (Xia et al. 2024)
    dict(name='ch3-agentless-abstract', arxiv_id='2407.01489', page=0, column='full',
         anchor='However, the complexity of these agent-based approaches',
         highlight=['Do we really have to employ complex autonomous software agents?', 'simplistic three-phase process of localization, repair, and patch validation',
                    'highest performance (32.00%, 96 correct fixes) and low cost ($0.70)'],
         above=4, end='open-source software agents!'),
    dict(name='ch3-agentless-figure1', arxiv_id='2407.01489', page=4, figure=True, column='full', anchor='Figure 1: Overview of AGENTLESS.', highlight=['Overview of AGENTLESS'], below=0),
    dict(name='ch3-agentless-table1', arxiv_id='2407.01489', page=9, column='full', anchor='SpecRover', above=20, highlight=['96 (32.00%)', '$0.70'],
         end='Performance on SWE-bench Lite'),
    # ---- DeepSeek-R1 (version 1, January 2025) and the o1 system card
    dict(name='ch3-r1-abstract', arxiv_id='2501.12948v1', page=0, column='full',
         anchor='DeepSeek-R1-Zero, a model trained via large-scale reinforcement learning',
         highlight=['without supervised fine-tuning (SFT) as a preliminary step', 'poor readability, and language mixing'],
         above=4, end='comparable to OpenAI-o1-1217 on reasoning tasks.'),
    dict(name='ch3-r1-aha', arxiv_id='2501.12948v1', page=7, column='full',
         anchor='A particularly intriguing phenomenon observed during the training of DeepSeek-R1-Zero',
         highlight=['allocate more thinking time to a problem by reevaluating its initial approach'], above=4, end='reevaluating its initial approach.'),
    dict(name='ch3-o1-training', arxiv_id='2412.16720', page=0, column='full',
         anchor='The o1 large language model family is trained with reinforcement learning to perform complex',
         highlight=['it can produce a long chain of thought before responding', 'refine their thinking process, try different strategies, and recognize their mistakes'],
         above=4, end='unsafe or inappropriate content.'),
]


def finish(names, trim=None):
    """Keep every picture at most 1100 px wide; trim the rotated arXiv stamp where it shows."""
    from PIL import Image
    trim = trim or {}
    for n in names:
        path = os.path.join(paper_shots.OUT, n + '.png')
        if not os.path.exists(path):
            continue
        im = Image.open(path)
        t = trim.get(n, 0)
        if t:
            im = im.crop((t, 0, im.width, im.height))
        if im.width > 1100:
            im = im.resize((1100, round(im.height * 1100 / im.width)), Image.LANCZOS)
        im.save(path, optimize=True)
        print(f'{n}: {im.width}x{im.height}')


TRIM = {'ch3-cot-figure1': 70, 'ch3-cot-abstract': 70, 'ch3-zeroshot-abstract': 70, 'ch3-unfaithful-abstract': 70, 'ch3-selfcorrect-abstract': 70,
        'ch3-ps-abstract': 0, 'ch3-rewoo-abstract': 70, 'ch3-critic-abstract': 70, 'ch3-tot-figure1': 0, 'ch3-sweagent-figure1': 70,
        'ch3-agentless-abstract': 70, 'ch3-r1-abstract': 70, 'ch3-o1-training': 70, 'ch3-sc-abstract': 70}

if __name__ == '__main__':
    only = os.environ.get('ONLY')
    done, failed = [], []
    for job in JOBS:
        if only and job['name'] not in only.split(','):
            continue
        try:
            shot(**job)
            done.append(job['name'])
        except (SystemExit, Exception) as e:          # keep going; report at the end
            failed.append((job['name'], str(e)[:160]))
    finish(done, TRIM)
    for n, e in failed:
        print('FAILED', n, e)
