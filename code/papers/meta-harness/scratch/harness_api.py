"""The fixed interface every harness must follow, and the cheap check run before any evaluation
(Algorithm 1, line 11: "if H passes interface validation"; Appendix D: "Lightweight validation").

A harness is one Python file that defines:

    class Harness:
        def __init__(self, llm, labels): ...   # llm(prompt, max_new_tokens=16) -> str ; labels: list of str
        def update(self, text, label): ...     # see one labeled example of the online stream
        def predict(self, text) -> str: ...    # return one of the labels

Rules: the harness may import `hlib` and the standard library, but it may not open files, run programs or use the
network. So it cannot read the test set, and everything it knows comes from update() and from the model.
"""
import importlib.util, re, sys, os, tempfile, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
FORBIDDEN = [r'\bopen\s*\(', r'\bimport\s+os\b', r'\bfrom\s+os\b', r'\bsubprocess\b', r'\bsocket\b', r'\burllib\b',
             r'\brequests\b', r'\b__import__\b', r'\beval\s*\(', r'\bexec\s*\(', r'task\.json', r'\bpathlib\b']


def load_module(path, name='candidate_harness'):
    """Import a harness file as a fresh module (hlib.py is found next to this file)."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    sys.dont_write_bytecode = True                  # no __pycache__ folders inside the candidate folders
    spec = importlib.util.spec_from_file_location(f'{name}_{abs(hash(path))}', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeLLM:
    """Stands in for the model during validation: always answers with the first label. Costs nothing."""
    def __init__(self, labels):
        self.labels = labels

    def __call__(self, prompt, max_new_tokens=16, **kw):
        assert isinstance(prompt, str), 'the harness must send a str prompt'
        return self.labels[0]


def validate(source, labels, examples):
    """Return (True, '') if the source follows the interface, else (False, reason). Takes well under a second."""
    for pat in FORBIDDEN:
        if re.search(pat, source):
            return False, f'forbidden pattern {pat!r}: a harness may not touch files, programs or the network'
    path = None
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(source)
            path = f.name
        mod = load_module(path, 'validate')
        if not hasattr(mod, 'Harness'):
            return False, 'no class named Harness'
        h = mod.Harness(FakeLLM(labels), list(labels))
        for ex in examples[:6]:
            h.update(ex['text'], ex['label'])
        for ex in examples[6:8]:
            out = h.predict(ex['text'])
            if out not in labels:
                return False, f'predict() returned {out!r}, which is not one of the labels'
    except Exception as e:                                        # any crash means the candidate is rejected
        tb = traceback.format_exc().strip().split('\n')
        return False, f'{type(e).__name__}: {e} | {tb[-3].strip() if len(tb) > 2 else ""}'
    finally:
        if path:
            os.unlink(path)
    return True, ''
