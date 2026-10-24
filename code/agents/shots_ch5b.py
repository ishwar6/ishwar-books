"""Highlighted excerpts for Chapter 5, second half (Sections 5.4 to 5.6): multi-agent debate (Du et al. 2023, Liang et al. 2023),
the later checks against voting (Smit et al. 2024, Wang et al. 2024, Choi et al. 2025), the MAST failure taxonomy (Cemri et al. 2025),
and passages of two engineering pages (Anthropic's multi-agent research system; the OpenAI Agents SDK docs), cut with blogshot.mjs.
Usage: python shots_ch5b.py   (ONLY=name1,name2 renders a subset; BLOGS=0 skips the web pages)."""
import os, subprocess
from paper_shots import shot, OUT
from shots_ch3 import finish

JOBS = [
    # ---- debate
    dict(name='ch5b-du-figure2', arxiv_id='2305.14325', page=2, figure=True, column='full', anchor='Figure 2: Illustration of Debate', highlight=[], below=0),
    dict(name='ch5b-du-table1', arxiv_id='2305.14325', page=5, column='full', anchor='Table 1: Multiagent Debate Improves Reasoning', above=120, below=26,
         highlight=['Multi-Agent (Majority)', 'Multi-Agent (Debate)']),
    dict(name='ch5b-du-agreeable', arxiv_id='2305.14325', page=3, anchor='Given multiple rounds of debate, how can we ensure', above=4,
         highlight=['led to longer debates and', 'relatively "agreeable"'], end='based on human feedback'),
    dict(name='ch5b-du-limits', arxiv_id='2305.14325', page=9, anchor='Finally, we found that while debates typically converged', above=4,
         highlight=['not necessarily correct', 'confidently affirm'], end='would improve the results'),
    dict(name='ch5b-liang-dot', arxiv_id='2305.19118', page=0, anchor='In this work, we focus on the Degeneration-of', above=4, below=150,
         highlight=['unable to generate novel thoughts later', 'even if the initial'], hl_nth={'unable to generate novel thoughts later': 1}),
    dict(name='ch5b-liang-figure2', arxiv_id='2305.19118', page=2, figure=True, column='full', anchor='Figure 2: Framework of Multi-Agent Debate', highlight=[], below=0),
    dict(name='ch5b-liang-table3', arxiv_id='2305.19118', page=4, anchor='Table 3: Accuracy on Counter', above=118, below=4, highlight=['+ MAD']),
    dict(name='ch5b-smit-abstract', arxiv_id='2311.17371', page=0, anchor='Recent advancements in large language models', above=4,
         highlight=['do not reliably outper', 'more sensitive to different hyperparameter'], end='protocols we evaluated.'),
    dict(name='ch5b-smit-figure1', arxiv_id='2311.17371', page=2, figure=True, column='full', anchor='Figure 1. Benchmark of experiment configurations on MedQA', highlight=[], below=26, fig_top=142),
    dict(name='ch5b-wang-abstract', arxiv_id='2402.18272', page=0, anchor='Recent progress in LLMs discussion suggests', above=4,
         highlight=['single-agent LLM with strong prompts', 'only when there', 'is no demonstration in the prompt'], end='during the discussion.'),
    dict(name='ch5b-choi-abstract', arxiv_id='2508.17536', page=0, column='full', anchor='Multi-Agent Debate (MAD) has emerged as a promising', above=4,
         highlight=['Majority Voting alone accounts for most of the performance gains', 'debate alone does not improve expected correctness'],
         end='in many practical settings.'),
    dict(name='ch5b-choi-figure2', arxiv_id='2508.17536', page=1, figure=True, column='full', anchor='Figure 2: Majority Voting is the main contributor', highlight=[], below=0),
    # ---- MAST
    dict(name='ch5b-mast-fc2', arxiv_id='2503.13657', page=6, column='full', anchor='FC2. Inter-Agent Misalignment. Failures arise', above=4,
         highlight=['withholding crucial information (FM-2.4, 0.85%)', 'mismatches between reasoning and action (FM-2.6, 13.2%)'],
         end="underscoring the need for"),
    dict(name='ch5b-mast-figure3', arxiv_id='2503.13657', page=5, figure=True, column='full', anchor='Figure 3: Visualization of a trace segment in MAST-Data', highlight=[], below=0),
    dict(name='ch5b-mast-fc3', arxiv_id='2503.13657', page=7, column='full', anchor='FC3 failures are related to the quality control', above=4,
         highlight=['the presence of a verifier is not a silver bullet', 'only superficial', '+15.6% improvement'], end='(details in Appendix H).'),
    dict(name='ch5b-mast-table5', arxiv_id='2503.13657', page=26, column='full', anchor='Table 5: Case Studies Accuracy Comparison', above=4, below=148,
         highlight=['New topology']),
]

# web pages: (name, url, text that selects the paragraph); blogshot.mjs captures the paragraph that contains the text
BLOGS = [
    ('ch5b-anthropic-think', 'https://www.anthropic.com/engineering/multi-agent-research-system', 'text=To iterate on prompts, you must understand their effects'),
    ('ch5b-anthropic-emergent', 'https://www.anthropic.com/engineering/multi-agent-research-system', 'text=small changes to the lead agent can unpredictably change'),
    ('ch5b-anthropic-judge', 'https://www.anthropic.com/engineering/multi-agent-research-system', 'text=We used an LLM judge that evaluated each output'),
    ('ch5b-anthropic-human', 'https://www.anthropic.com/engineering/multi-agent-research-system', 'text=People testing agents find edge cases that evals miss'),
    ('ch5b-anthropic-stateful', 'https://www.anthropic.com/engineering/multi-agent-research-system', 'text=Agents can run for long periods of time, maintaining state'),
    ('ch5b-anthropic-debugging', 'https://www.anthropic.com/engineering/multi-agent-research-system', 'text=Adding full production tracing let us diagnose'),
    ('ch5b-anthropic-rainbow', 'https://www.anthropic.com/engineering/multi-agent-research-system', 'text=we use rainbow deployments to avoid disrupting'),
    ('ch5b-sdk-maxturns', 'https://openai.github.io/openai-agents-python/running_agents/', 'text=If we exceed the max_turns passed'),
    ('ch5b-sdk-tracing', 'https://openai.github.io/openai-agents-python/tracing/', 'text=Handoffs are wrapped in'),
]

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
    if os.environ.get('BLOGS', '1') != '0':
        here = os.path.dirname(os.path.abspath(__file__))
        for name, url, sel in BLOGS:
            if only and name not in only.split(','):
                continue
            extra = ['--up', '1', '--pad', '8'] if name in ('ch5b-sdk-maxturns', 'ch5b-sdk-tracing') else []
            r = subprocess.run(['node', os.path.join(here, 'blogshot.mjs'), url, os.path.join(OUT, name + '.png'), sel, '--width', '1000'] + extra,
                               capture_output=True, text=True, timeout=120)
            (done.append(name) if r.returncode == 0 else failed.append((name, (r.stderr or r.stdout)[-200:])))
            print(r.stdout.strip())
    finish(done, {'ch5b-choi-abstract': 200})
    for n, e in failed:
        print('FAILED', n, e)
