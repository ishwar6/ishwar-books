"""Evaluate(H, M, X) of Algorithm 1: run one harness on one set of messages and record everything.

The protocol is the online one of §4.1: the harness first sees the labeled memory stream through update(), one
example at a time, then it predicts each message of the evaluation set. Every model call is recorded (the prompt and
the reply), so the proposer can later read exactly what the model saw and said."""
import collections, time
from harness_api import load_module


class TracingLLM:
    """Wraps the frozen model. Records each call of the current message, so the trace shows every prompt."""
    def __init__(self, llm):
        self.llm, self.calls = llm, []

    def __call__(self, prompt, max_new_tokens=16, **kw):
        max_new_tokens = min(int(max_new_tokens), 32)          # a label is short; a harness may not ask for more
        reply = self.llm(prompt, max_new_tokens=max_new_tokens)
        self.calls.append({'prompt': prompt, 'reply': reply})
        return reply


def evaluate(harness_path, llm, labels, memory, eval_set):
    """Return (scores, traces). Accuracy is the share of messages whose predicted label equals the gold label."""
    t0 = time.time()
    tracer = TracingLLM(llm)
    harness = load_module(harness_path).Harness(tracer, list(labels))
    for ex in memory:
        harness.update(ex['text'], ex['label'])
    traces, correct, invalid, errors = [], 0, 0, 0
    per_label = collections.defaultdict(lambda: [0, 0])
    confusions = collections.Counter()
    for i, ex in enumerate(eval_set):
        tracer.calls = []
        try:
            pred = harness.predict(ex['text'])
            err = None
        except Exception as e:                                  # a crash on one message counts as a wrong answer
            pred, err = None, f'{type(e).__name__}: {e}'
            errors += 1
        if pred not in labels:
            invalid += 1
        ok = pred == ex['label']
        correct += ok
        per_label[ex['label']][0] += ok
        per_label[ex['label']][1] += 1
        if not ok:
            confusions[f"{ex['label']} -> {pred}"] += 1
        traces.append({'id': i, 'text': ex['text'], 'gold': ex['label'], 'pred': pred, 'correct': ok,
                       'calls': tracer.calls, 'prompt_chars': sum(len(c['prompt']) for c in tracer.calls),
                       **({'error': err} if err else {})})
    n = len(eval_set)
    scores = {
        'accuracy': round(100 * correct / n, 1), 'correct': correct, 'n': n,
        'ctx_chars': round(sum(t['prompt_chars'] for t in traces) / n),      # prompt characters per message
        'calls_per_message': round(sum(len(t['calls']) for t in traces) / n, 2),
        'invalid': invalid, 'crashes': errors,
        'accuracy_by_label': {l: f'{c}/{m}' for l, (c, m) in sorted(per_label.items())},
        'top_confusions': dict(confusions.most_common(5)),
        'seconds': round(time.time() - t0, 1),
    }
    return scores, traces
