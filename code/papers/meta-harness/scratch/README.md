# Meta-Harness, from scratch

A small, complete version of the search loop of "Meta-Harness: End-to-End Optimization of Model Harnesses"
(Lee et al., 2026, arXiv 2603.28052), on a laptop, with small local models and no API.
The page that walks through it: https://ishwarj.com/papers/meta-harness/part-6-build-it-from-scratch

- **Task**: label a bank customer's message with one of 10 card intents (a slice of Banking77, one of the paper's
  out-of-distribution datasets). Memory stream 60 labeled messages, search set 50, test set 100.
- **Frozen task model M**: `Qwen/Qwen2.5-0.5B-Instruct`, greedy decoding.
- **Harness**: one Python file with `class Harness` and the methods `update(text, label)` and `predict(text)`.
- **Filesystem D**: one folder per candidate with `harness.py`, `scores.json`, `traces.jsonl`, `proposal.md`.
- **Proposers**: a scripted one (rules over a design space), a local LLM (`Qwen/Qwen2.5-1.5B-Instruct`), and two
  that need a key or an agent and are not run here (an OpenAI-compatible chat API, a command-line coding agent).

| File | What it is |
|---|---|
| `task.py` | The data: 10 Banking77 intents, the memory, search and test sets (`data/task.json`). |
| `llm.py` | The frozen task model: `LocalLLM` (transformers `generate`) and `greedy_from_scratch`. |
| `hlib.py` | Helpers a harness may import: TF-IDF retrieval written from scratch, `parse_label`. |
| `harness_api.py` | The harness interface and `validate()`, the cheap check before any evaluation. |
| `design_space.py` | A harness template with 7 knobs (examples, retrieval, coverage, contrastive pairs, draft verification...). The seeds come from it. |
| `evaluate.py` | `evaluate()`: the online protocol (update on the memory, then predict), with a trace of every model call. |
| `store.py` | The filesystem of candidates, and the Pareto frontier (two ways). |
| `proposers.py` | `ScriptedProposer`, `LocalLLMProposer`, `ChatAPIProposer`, `AgentCLIProposer`. |
| `skill.md` | The instructions the LLM proposers get (Appendix D: "Write a good skill"). |
| `loop.py` | Algorithm 1, line by line. |
| `run.py` | One full run: seeds, search, Pareto frontier, then one test-set evaluation. |
| `mh.py` | A small CLI over a run folder: `list`, `frontier`, `errors`, `diff`, `test`. |
| `runs/` | The saved runs (every candidate's code, scores and traces). |

## Run it

You need Python 3.10+, `torch` and `transformers` (the models download once, about 1 GB and 3 GB).

```bash
cd code/papers/meta-harness/scratch
python run.py --proposer scripted  --iters 5 --k 2 --name scripted-1     # 35 minutes on a shared M-series laptop GPU
python run.py --proposer local-llm --iters 5 --k 2 --name local-llm-1 --test frontier   # the 1.5B model writes the harnesses
python mh.py runs/scripted-1 frontier
python mh.py runs/scripted-1 test
MH_SMOKE=1 python run.py --iters 1 --name smoke                         # 6 search messages: tests the code in a minute
```

With a key for an OpenAI-compatible chat endpoint (not run for the page):

```bash
export MH_API_BASE=https://<provider>/v1 MH_API_KEY=... MH_API_MODEL=<model>
python run.py --proposer api --iters 10 --k 2 --name api-1
```

With a coding agent that has a non-interactive mode (the paper's setup; not run for the page):

```bash
export MH_AGENT_CMD='<your-agent-command> --print {prompt}'
python run.py --proposer agent-cli --iters 10 --k 2 --name agent-1
```

## Rules that keep the test honest

- Proposers read only `candidates/`. Test results go to `heldout/`, and only after the search ends.
- A harness may not open files, import `os`, run programs or use the network (`harness_api.FORBIDDEN`), so it cannot
  read `data/task.json`. Everything it knows comes from `update()` and from the model.
- The candidate chosen for the final number is the one with the best **search** accuracy, never the best test accuracy.

## Write your own harness

```python
from hlib import TfIdf, parse_label

class Harness:
    def __init__(self, llm, labels):
        self.llm, self.labels, self.memory = llm, labels, []

    def update(self, text, label):
        self.memory.append((text, label))

    def predict(self, text):
        index = TfIdf([t for t, _ in self.memory])
        shots = [self.memory[i] for i in index.top(text, 4)]
        prompt = 'Labels: ' + ', '.join(self.labels) + '\n\n'
        prompt += ''.join(f'Message: {t}\nLabel: {l}\n\n' for t, l in shots)
        prompt += f'Message: {text}\nLabel:'
        return parse_label(self.llm(prompt), self.labels) or self.labels[0]
```

Save it as `my_harness.py` and score it:

```python
import task, llm
from evaluate import evaluate
labels, memory, search, test = task.load()
scores, traces = evaluate('my_harness.py', llm.LocalLLM(), labels, memory, search)
print(scores['accuracy'], scores['ctx_chars'])
```
