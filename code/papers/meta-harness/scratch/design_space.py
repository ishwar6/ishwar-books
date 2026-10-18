"""A design space of classification harnesses: one Python template with a CONFIG dict of knobs.

The scripted proposer (proposers.py) searches this space. The seeds are points in it too. Every rendered harness is a
complete, readable Python file, so the local LLM proposer can read it and change the code itself, not only the knobs.

The knobs, and the strategy of the paper each one echoes (Appendix B.1, Figures 5 and 6):
  shots        how many labeled examples go into the prompt (0 = zero-shot)
  select       which examples: 'first' (the first ones seen), 'random', or 'tfidf' (most similar to the message)
  coverage     add one most-similar example for every label (the "coverage block" of the label-primed harness)
  contrastive  add similar examples whose labels differ from each other (the "contrastive pairs")
  verify       two calls: a draft label, then a check against confirmers and challengers (draft verification)
  label_style  'snake' (card_arrival) or 'words' (card arrival)
  instruction  'short', or 'careful' (warns that the labels are close)
"""
import ast, json, re

DEFAULT = dict(shots=0, select='first', coverage=False, contrastive=False, verify=False,
               label_style='snake', instruction='short')
CHOICES = dict(shots=[0, 4, 8, 16, 60], select=['first', 'random', 'tfidf'], coverage=[False, True],
               contrastive=[False, True], verify=[False, True], label_style=['snake', 'words'],
               instruction=['short', 'careful'])

TEMPLATE = '''"""Harness {name}: {about}"""
import random
from hlib import TfIdf, parse_label, show_label

CONFIG = {config}

INSTRUCTIONS = {{
    'short': 'Classify the bank customer message into exactly one of the labels.',
    'careful': ('Classify the bank customer message into exactly one of the labels. Several labels are close: '
                'pick the one that names the exact problem in the message.'),
}}


class Harness:
    def __init__(self, llm, labels):
        self.llm, self.labels = llm, labels
        self.memory = []                     # every (text, label) pair seen so far, in order
        self.index = None                    # TF-IDF index over the memory, built when first needed

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
        return [self.nearest(text, 1, keep=lambda l, x=label: l == x)[0]
                for label in self.labels if any(l == label for _, l in self.memory)]

    def contrastive_block(self, text):
        out, seen = [], set()
        for t, l in self.nearest(text, 8):   # the 8 nearest; keep the first example of each label
            if l not in seen:
                seen.add(l)
                out.append((t, l))
        return out[:4]

    def prompt(self, text, shown, extra=''):
        lines = [INSTRUCTIONS[CONFIG['instruction']], '', 'Labels:']
        lines += ['- ' + self.name(l) for l in self.labels]
        if shown:
            lines += ['', 'Examples:']
            lines += [f'Message: {{t}}\\nLabel: {{self.name(l)}}' for t, l in shown]
        if extra:
            lines += ['', extra]
        lines += ['', f'Message: {{text}}', 'Answer with the label only.', 'Label:']
        return '\\n'.join(lines)

    # ---- the two methods the evaluator calls -----------------------------------------------------
    def predict(self, text):
        shown = self.examples(text)
        if CONFIG['coverage'] and self.memory:
            cover = self.coverage_block(text)
            shown = cover + [e for e in shown if e not in cover]
        if CONFIG['contrastive'] and self.memory:
            shown = shown + [e for e in self.contrastive_block(text) if e not in shown]
        draft = parse_label(self.llm(self.prompt(text, shown)), self.labels) or self.labels[0]
        if not CONFIG['verify'] or len(self.memory) < 5:
            return draft
        confirm = self.nearest(text, 3, keep=lambda l: l == draft)
        challenge = self.nearest(text, 3, keep=lambda l: l != draft)
        note = (f'A first answer was "{{self.name(draft)}}". Check it. Examples with that label and similar '
                'examples with other labels are shown above. Keep it or change it.')
        final = self.llm(self.prompt(text, confirm + challenge, extra=note))
        return parse_label(final, self.labels) or draft
'''


def render(name, cfg, about=''):
    cfg = {**DEFAULT, **cfg}
    return TEMPLATE.format(name=name, about=about or describe(cfg), config=repr(cfg))


def describe(cfg):
    cfg = {**DEFAULT, **cfg}
    parts = ['zero-shot' if cfg['shots'] == 0 else f"{cfg['shots']} {cfg['select']} examples"]
    parts += [k for k in ('coverage', 'contrastive', 'verify') if cfg[k]]
    if cfg['label_style'] != 'snake':
        parts.append('labels as words')
    if cfg['instruction'] != 'short':
        parts.append('careful instruction')
    return ', '.join(parts)


def read_config(source):
    """Find the CONFIG dict in a harness file (None if the file has no CONFIG of this design space)."""
    m = re.search(r'^CONFIG = (\{.*\})\s*$', source, re.M)
    if not m:
        return None
    try:
        cfg = ast.literal_eval(m.group(1))
    except (ValueError, SyntaxError):
        return None
    if not isinstance(cfg, dict):
        return None
    return {**DEFAULT, **cfg} if set(cfg) <= set(DEFAULT) else None


def key(cfg):
    return json.dumps({**DEFAULT, **cfg}, sort_keys=True)


def neighbours(cfg):
    """Every config that differs from cfg in exactly one knob."""
    cfg = {**DEFAULT, **cfg}
    for k, values in CHOICES.items():
        for v in values:
            if v != cfg[k]:
                yield k, v, {**cfg, k: v}


SEEDS = {
    'zero_shot': dict(shots=0),
    'few_shot_8': dict(shots=8, select='random'),
    'few_shot_all': dict(shots=60, select='first'),
}
