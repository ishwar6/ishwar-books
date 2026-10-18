# Proposal (local LLM proposer Qwen/Qwen2.5-1.5B-Instruct, iteration 1, sample 0)

IDEA: (none given)

Files it asked for: '002_few_shot_all/traces.jsonl'
Files read: ['002_few_shot_all/traces.jsonl']; also shown: 001_few_shot_8/harness.py
Self-check attempts: ['ImportError: cannot import name \'log\' from \'hlib\' (/Users/admin/Desktop/ishwar-books/code/papers/meta-harness/scratch/hlib.py) | File "/var/folders/9d/nfsy2j656v107w7r9bpvf7440000gn/T/tmpjzd0kbtu.py", line 2, in <module>', 'ImportError: cannot import name \'log\' from \'hlib\' (/Users/admin/Desktop/ishwar-books/code/papers/meta-harness/scratch/hlib.py) | File "/var/folders/9d/nfsy2j656v107w7r9bpvf7440000gn/T/tmpt9pa4wck.py", line 2, in <module>']

## Full reply

```python
import random
from hlib import TfIdf, parse_label, show_label, log

CONFIG = {'shots': 8, 'select': 'random', 'coverage': True, 'contrastive': True, 'verify': False, 'label_style': 'snake'}
INSTRUCTIONS = {
    'short': 'Classify the bank customer message into exactly one of the labels.',
    'careful': ('Classify the bank customer message into exactly one of the labels. Several labels are close: '
                'pick the one that names the exact problem in the message.'),
}

class Harness:
    def __init__(self, llm, labels):
        self.llm, self.labels = llm, labels
        self.memory = []                         # every (text, label) pair seen so far, in order
        self.index = None                        # TF-IDF index over the memory, built when first needed

    def update(self, text, label):
        """See one labeled example (the online stream). Store it; rebuild the index later."""
        self.memory.append((text, label))
        self.index = None

    # ---- helpers ------------------------------------------------------------------------------
    def name(self, label):
        return show_label(label) if CONFIG['label_style'] == 'words' else label

    def nearest(self, text, k, keep=lambda label: True):
        """The k stored examples most similar to text (TF-IDF cosine) whose label passes keep()."""
        if self.index is None:
            self.index = TfIdf([t for t, _ in self.memory])
        order = self.index.top(text, len(self.memory))
        return [self.memory[i] for i in order if keep(self.memory[i][1])][:k]

    def examples(self, text):
        k, how = CONFIG['shots'], CONFIG['select']
        if k == 0 or not self.memory:
            return []
        if how == 'first':
            return self.memory[:k]
        if how == 'random':
            return random.Random(0).sample(self.memory, min(k, len(self.memory)))
        return self.nearest(text, k)[::-1]   # tfidf: most similar example last, next to the question

    def coverage_block(self, text):
        return [self.nearest(text, 1, keep=lambda l: l == x)[0] for x in self.labels]

    def contrastive_block(self, text):
        out, seen = [], set()
        for t, l in self.nearest(text, 8):   # the 8 nearest; keep the first example of each label
            if l not in seen:
                seen.add(l)
                out.append((t, l))
        return out[:4]

    def prompt(self, text, shown, extra=''):
        lines = [INSTRUCTIONS[CONFIG['instruction']], '', 'Labels:', ]
        lines += ['-' * 25 + '-' * 12 + '-' * len(x) + ' ' + y + ':' for x, y in zip(self.labels, self.names)]
        lines += ['', 'Examples:', ]
        lines += [[f'Message: {t}\nLabel: {x}' for t, x in shown]]
        if extra:
            lines += ['', extra]
        lines += ['', f'Message: {text}', 'Answer with the label only.', 'Label:']
        return '\n'.join(lines)

    # ---- the two methods the evaluator calls -----------------------------------------------------
    def predict(self, text):
        shown = self.examples(text)
        if CONFIG['coverage']:
            cover = [self.coverage_block(text)[i] for i in range(len(self.labels)) if i == 0]
            shown = cover + [e for e in shown if e not in cover]
        draft = self.llm(self.prompt(text, shown, ''))[0]
        if not CONFIG['verify'] or len(self.memory) < 5:
            return draft
        confirm = [e for e in self.contrastive_block(text) if e not in shown]
        challenge = [e for e in shown if e not in self.contrastive_block(text)]
        note = (f'A first answer was `{draft}`. Check it. Examples with that label and similar examples with other labels are shown above. Keep it or change it.')
        final = self.llm(self.prompt(text, confirm + challenge, note))[0]
        return parse_label(final, self.labels) or draft
```
