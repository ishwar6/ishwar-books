"""Zero-shot: ask for the label, show no examples."""

def parse(out):
    """Read the model's answer: 'positive', 'negative', or None if it is not clear."""
    o = out.lower()
    pos, neg = 'positive' in o, 'negative' in o
    return 'positive' if pos and not neg else 'negative' if neg and not pos else None


def run(x, model, memory):
    prompt = f"Review: {x}\nIs the sentiment of this review positive or negative?"
    out = model(prompt)
    return parse(out), [dict(step='answer', prompt=prompt, output=out)]
