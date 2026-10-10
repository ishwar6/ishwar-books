"""Small helpers shared by the ViT scripts: a log that is printed and saved, and JSON results."""
import json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
os.makedirs(RESULTS, exist_ok=True)


class Log:
    """print() that also writes every line to results/<name>_stdout.txt (the text behind the terminal screenshots)."""
    def __init__(self, name):
        self.path = os.path.join(RESULTS, f'{name}_stdout.txt')
        self.lines = []

    def __call__(self, *args):
        line = ' '.join(str(a) for a in args)
        print(line)
        self.lines.append(line)
        open(self.path, 'w').write('\n'.join(self.lines) + '\n')


def save(name, data):
    path = os.path.join(RESULTS, f'{name}.json')
    json.dump(data, open(path, 'w'), indent=1)
    return path


def timer():
    t0 = time.time()
    return lambda: time.time() - t0
