"""Chapter 6, second half: terminal screenshots of the saved runs of ch6b_judge.py and ch6b_trajectory.py (termshot.py renders
results/<name>_stdout.txt as public/img/agents/<name>-run.png). Usage: python ch6b_termshots.py"""
from termshot import render

for n in ['ch6b_judge', 'ch6b_trajectory']:
    render(n, title=f'python {n}.py', max_lines=70)
