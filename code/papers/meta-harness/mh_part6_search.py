"""Part 6: run the from-scratch Meta-Harness search (scratch/run.py) and save its printed log to results/.

  python mh_part6_search.py scripted     -> scratch/runs/scripted-1,  results/part6_search_scripted_stdout.txt
  python mh_part6_search.py local-llm    -> scratch/runs/local-llm-1, results/part6_search_local-llm_stdout.txt
Each run: 3 seeds + 5 iterations x 2 proposals, then the test set."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch'))
from common import Log
import run

if __name__ == '__main__':
    which = sys.argv[1]
    test = 'all' if which == 'scripted' else 'frontier'
    run.main(['--proposer', which, '--iters', '5', '--k', '2', '--name', f'{which}-1', '--test', test],
             echo=Log(f'part6_search_{which}'))
