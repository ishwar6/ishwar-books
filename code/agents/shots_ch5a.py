"""Highlighted excerpts for Chapter 5, first half (multi-agent systems): CAMEL, MetaGPT, ChatDev and AutoGen from their arXiv PDFs,
and passages of engineering posts and specifications captured from the live pages with blogshot.mjs (Anthropic's multi-agent research
system, Cognition's "Don't Build Multi-Agents", OpenAI Swarm and the Agents SDK, the MCP and A2A specifications).
Usage: python shots_ch5a.py  (ONLY=name1,name2 renders a subset)  ->  public/img/agents/ch5a-*.png"""
import os, subprocess
import paper_shots
from paper_shots import shot
from shots_ch3 import finish

HERE = os.path.dirname(os.path.abspath(__file__))

PAPERS = [
    # ---- CAMEL (Li et al., 2023): role-playing between an AI user and an AI assistant
    dict(name='ch5a-camel-figure1', arxiv_id='2303.17760', page=3, figure=True, column='full',
         anchor='Figure 1: CAMEL Role-Playing Framework', highlight=[], below=0),
    dict(name='ch5a-camel-challenges', arxiv_id='2303.17760', page=7, column='full', anchor='Role Flipping: One challenge we encountered',
         highlight=['the assistant and user switch roles', 'infinite loop of meaningless conversation', 'unable to break out of it'],
         above=4, end='are unable to break out of it.'),
    # ---- MetaGPT (Hong et al., 2023): roles, standard operating procedures and a shared message pool
    dict(name='ch5a-metagpt-pool', arxiv_id='2308.00352', page=5, column='full', anchor='Sharing information is critical in collaboration',
         highlight=['store information in a global message pool', 'Sharing all information with every agent can lead to information overload',
                    'select information to follow based on their role profiles'], above=4, end='might be of lesser concern.'),
    dict(name='ch5a-metagpt-chatter', arxiv_id='2308.00352', page=1, column='full', anchor='More graphically, in a company simulated by MetaGPT',
         highlight=['all their handovers must comply with certain established standards', 'idle chatter between LLMs'], above=4,
         end='Bob (Architect).'),
    # ---- ChatDev (Qian et al., 2023): a chat chain of role pairs, passing only solutions between phases
    dict(name='ch5a-chatdev-figure2', arxiv_id='2307.07924', page=2, figure=True, column='full',
         anchor='Figure 2: Upon receiving a preliminary task requirement', highlight=[], below=0),
    dict(name='ch5a-chatdev-memory', arxiv_id='2307.07924', page=3, anchor='By sharing only the solutions of each subtask',
         highlight=['sharing only the solutions of each subtask rather than the entire communication history'], above=4,
         end='context continuity.'),
    # ---- AutoGen (Wu et al., 2023): conversation patterns, and the single-agent ablation
    dict(name='ch5a-autogen-ablation', arxiv_id='2308.08155', page=7, column='full', anchor='We also conduct an ablation showing that multi-agent abstraction is necessary',
         highlight=['a single agent conducts both the code-writing and safeguard processes', 'by 8% (with GPT-4) and 35% (with GPT-3.5-turbo)'],
         above=4, end='35% (with GPT-3.5-turbo).'),
]

ANTH = 'https://www.anthropic.com/engineering/built-multi-agent-research-system'
COG = 'https://cognition.com/blog/dont-build-multi-agents'
BLOGS = [  # name, url, target, extra args
    ('ch5a-anthropic-compression', ANTH, 'text=The essence of search is compression', []),
    ('ch5a-anthropic-tokens', ANTH, 'text=Multi-agent systems work mainly because they help spend enough tokens', []),
    ('ch5a-anthropic-effort', ANTH, 'text=Scale effort to query complexity', []),
    ('ch5a-anthropic-sync', ANTH, 'text=Synchronous execution creates bottlenecks', []),
    ('ch5a-anthropic-telephone', ANTH, "text=Direct subagent outputs can bypass the main coordinator", []),
    ('ch5a-cognition-flappy', COG, 'text=Suppose your Task is', []),
    ('ch5a-cognition-principle1', COG, 'text=Share context, and share full agent traces', []),
    ('ch5a-cognition-principle2', COG, 'text=Actions carry implicit decisions, and conflicting decisions', []),
    ('ch5a-sdk-handoff-tools', 'https://openai.github.io/openai-agents-python/handoffs/', 'text=Handoffs are represented as tools to the LLM', []),
    ('ch5a-sdk-handoff-history', 'https://openai.github.io/openai-agents-python/handoffs/', 'text=gets to see the entire previous conversation history', []),
    ('ch5a-sdk-patterns', 'https://openai.github.io/openai-agents-python/multi_agent/', 'table', []),
    ('ch5a-mcp-overview', 'https://modelcontextprotocol.io/specification/latest', 'text=Hosts: LLM applications that initiate connections', ['--up', '1', '--pad', '10']),
    ('ch5a-mcp-features', 'https://modelcontextprotocol.io/specification/latest', 'text=Functions for the AI model to execute', ['--up', '1', '--pad', '10']),
    ('ch5a-a2a-intro', 'https://a2a-protocol.org/latest/specification/', 'text=Opaque Execution: Agents collaborate based on declared capabilities', ['--pad', '6']),
]


def blog(name, url, target, extra):
    out = os.path.join(paper_shots.OUT, name + '.png')
    pad = [] if '--pad' in extra else ['--pad', '18']
    r = subprocess.run(['node', os.path.join(HERE, 'blogshot.mjs'), url, out, target, '--width', '1000', *pad, *extra],
                       capture_output=True, text=True, timeout=180)
    print((r.stdout + r.stderr).strip()[-300:])
    return r.returncode == 0


if __name__ == '__main__':
    only = os.environ.get('ONLY')
    done, failed = [], []
    for job in PAPERS:
        if only and job['name'] not in only.split(','):
            continue
        try:
            shot(**job)
            done.append(job['name'])
        except (SystemExit, Exception) as e:
            failed.append((job['name'], str(e)[:160]))
    for name, url, target, extra in BLOGS:
        if only and name not in only.split(','):
            continue
        if blog(name, url, target, extra):
            done.append(name)
        else:
            failed.append((name, 'blogshot failed'))
    finish(done, {'ch5a-chatdev-memory': 0})
    for n, e in failed:
        print('FAILED', n, e)
