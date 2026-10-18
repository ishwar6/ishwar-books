"""The harness files of Part 2. Each one is the full text of a single-file Python harness, as the paper uses.
The interface (checked by `validate` in mh_part2.py): a function run(x, model, memory) -> (label, trace).
  x      : the review text to classify
  model  : a function prompt -> text (the frozen language model M)
  memory : a list of (text, label) pairs the harness may use (labelled examples)
  label  : 'positive', 'negative' or None;  trace: a list of dicts, one per model call
The seeds are the two baselines.  The other files are the stub proposer's design space."""

PARSE = '''
def parse(out):
    """Read the model's answer: 'positive', 'negative', or None if it is not clear."""
    o = out.lower()
    pos, neg = 'positive' in o, 'negative' in o
    return 'positive' if pos and not neg else 'negative' if neg and not pos else None
'''

SEEDS = {
    'zero_shot': '''"""Zero-shot: ask for the label, show no examples."""
''' + PARSE + '''

def run(x, model, memory):
    prompt = f"Review: {x}\\nIs the sentiment of this review positive or negative?"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
''',
    'few_shot_4': '''"""Few-shot: show the first 4 labelled examples of memory, then the review."""
''' + PARSE + '''

def run(x, model, memory):
    shots = ''.join(f"Review: {t}\\nSentiment: {y}\\n\\n" for t, y in memory[:4])
    prompt = shots + f"Review: {x}\\nSentiment:"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
''',
}

# The stub proposer's design space: each entry is (name, the problem in the traces it answers, the file).
EDITS = [
    ('one_word_answer', 'answers that could not be parsed', '''"""Zero-shot, but the prompt fixes the answer format: one word."""
''' + PARSE + '''

def run(x, model, memory):
    prompt = (f"Review: {x}\\nIs the sentiment of this review positive or negative?\\n"
              "Answer with exactly one word: positive or negative.")
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
'''),
    ('broken_rename', 'a refactor that renames the entry point', '''"""A broken candidate: the entry point is renamed, so the interface check must reject it."""
''' + PARSE + '''

def classify(x, model, memory):
    return parse(model(x)), []
'''),
    ('retrieve_4', 'wrong labels on reviews unlike the 4 fixed examples', '''"""Few-shot with retrieval: show the 4 memory examples that share the most words with the review."""
''' + PARSE + '''

def words(s):
    return set(s.lower().split())


def run(x, model, memory):
    q = words(x)
    ranked = sorted(memory, key=lambda m: -len(q & words(m[0])) / len(q | words(m[0])))   # Jaccard overlap
    shots = ''.join(f"Review: {t}\\nSentiment: {y}\\n\\n" for t, y in ranked[:4])
    prompt = shots + f"Review: {x}\\nSentiment:"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out, shown=[t[:40] for t, _ in ranked[:4]])]
'''),
    ('retrieve_balanced', 'retrieved examples that all carry one label', '''"""Retrieval, balanced: the 2 most similar positive and the 2 most similar negative examples, label list first."""
''' + PARSE + '''

def words(s):
    return set(s.lower().split())


def run(x, model, memory):
    q = words(x)
    sim = lambda m: len(q & words(m[0])) / len(q | words(m[0]))                            # Jaccard overlap
    pos = sorted([m for m in memory if m[1] == 'positive'], key=lambda m: -sim(m))[:2]
    neg = sorted([m for m in memory if m[1] == 'negative'], key=lambda m: -sim(m))[:2]
    shots = ''.join(f"Review: {t}\\nSentiment: {y}\\n\\n" for t, y in pos + neg)
    prompt = "Labels: positive, negative\\n\\n" + shots + f"Review: {x}\\nSentiment:"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
'''),
]
