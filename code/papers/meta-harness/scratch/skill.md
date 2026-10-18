# Skill: improve a text-classification harness

You are the proposer of a harness search. Your job: write ONE new harness file that classifies bank customer
messages more accurately than the harnesses tried so far, or as accurately with a shorter prompt.

## What a harness is

A Python file that defines `class Harness` with three methods:

- `__init__(self, llm, labels)`: `llm(prompt, max_new_tokens=16)` returns the model's reply as a string.
  `labels` is the list of valid labels.
- `update(self, text, label)`: called once for each labeled example of the memory stream, before any prediction.
- `predict(self, text)`: must return one string from `labels`.

You may import `hlib` (`TfIdf`, `parse_label`, `show_label`, `words`) and the standard library.
You may NOT open files, import `os`, run programs or use the network. A file that breaks a rule is rejected.

## Where things are

- `candidates/<NNN_name>/harness.py`: the code of every harness tried so far.
- `candidates/<NNN_name>/scores.json`: its accuracy on the search set (%), `ctx_chars` (prompt characters per
  message, lower is cheaper), the accuracy for each label and the most common confusions.
- `candidates/<NNN_name>/traces.jsonl`: one JSON line per search message, with every prompt the harness sent,
  the model's reply, the predicted label and the gold label.
- `candidates/<NNN_name>/proposal.md`: why that harness was proposed.
- The test set is not in this folder. You never see test results.

## How to work

1. Read the scores of all candidates. Look at the best ones and at the ones that regressed.
2. Read the traces of wrong answers. Find why the model chose the wrong label: what did the prompt show it?
3. Make a hypothesis, then write a new harness that tests it. Change one idea at a time when you can.

## Output

First one line that starts with `IDEA:` and says what you changed and why. Then the complete file in one
```python code block. Do not leave out any part of the file.
