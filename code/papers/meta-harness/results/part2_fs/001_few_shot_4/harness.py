"""Few-shot: show the first 4 labelled examples of memory, then the review."""

def parse(out):
    """Read the model's answer: 'positive', 'negative', or None if it is not clear."""
    o = out.lower()
    pos, neg = 'positive' in o, 'negative' in o
    return 'positive' if pos and not neg else 'negative' if neg and not pos else None


def run(x, model, memory):
    shots = ''.join(f"Review: {t}\nSentiment: {y}\n\n" for t, y in memory[:4])
    prompt = shots + f"Review: {x}\nSentiment:"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
