"""Retrieval, balanced: the 2 most similar positive and the 2 most similar negative examples, label list first."""

def parse(out):
    """Read the model's answer: 'positive', 'negative', or None if it is not clear."""
    o = out.lower()
    pos, neg = 'positive' in o, 'negative' in o
    return 'positive' if pos and not neg else 'negative' if neg and not pos else None


def words(s):
    return set(s.lower().split())


def run(x, model, memory):
    q = words(x)
    sim = lambda m: len(q & words(m[0])) / len(q | words(m[0]))                            # Jaccard overlap
    pos = sorted([m for m in memory if m[1] == 'positive'], key=lambda m: -sim(m))[:2]
    neg = sorted([m for m in memory if m[1] == 'negative'], key=lambda m: -sim(m))[:2]
    shots = ''.join(f"Review: {t}\nSentiment: {y}\n\n" for t, y in pos + neg)
    prompt = "Labels: positive, negative\n\n" + shots + f"Review: {x}\nSentiment:"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
