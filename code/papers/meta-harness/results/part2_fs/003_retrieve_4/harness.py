"""Few-shot with retrieval: show the 4 memory examples that share the most words with the review."""

def parse(out):
    """Read the model's answer: 'positive', 'negative', or None if it is not clear."""
    o = out.lower()
    pos, neg = 'positive' in o, 'negative' in o
    return 'positive' if pos and not neg else 'negative' if neg and not pos else None


def words(s):
    return set(s.lower().split())


def run(x, model, memory):
    q = words(x)
    ranked = sorted(memory, key=lambda m: -len(q & words(m[0])) / len(q | words(m[0])))   # Jaccard overlap
    shots = ''.join(f"Review: {t}\nSentiment: {y}\n\n" for t, y in ranked[:4])
    prompt = shots + f"Review: {x}\nSentiment:"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out, shown=[t[:40] for t, _ in ranked[:4]])]
