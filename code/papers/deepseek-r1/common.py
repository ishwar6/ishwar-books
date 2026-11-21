"""Small helpers shared by the DeepSeek-R1 scripts: a log that is printed and saved, JSON results, and a lock for heavy jobs."""
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


class heavy:
    """Lock for jobs that load a model or train (several scripts share one Mac).
    Usage: `with heavy('part4'):` ... Waits until no other heavy job runs. Never time code outside this lock."""
    LOCK = os.path.join(RESULTS, '.heavy.lock')

    def __init__(self, who):
        self.who = who

    def __enter__(self):
        waited = 0
        while True:
            try:
                os.mkdir(self.LOCK)
                open(os.path.join(self.LOCK, 'owner'), 'w').write(f'{self.who} pid {os.getpid()}')
                return self
            except FileExistsError:
                try:   # a lock left by a dead process is removed
                    pid = int(open(os.path.join(self.LOCK, 'owner')).read().split()[-1])
                    os.kill(pid, 0)
                except (OSError, ValueError, IndexError):
                    import shutil; shutil.rmtree(self.LOCK, ignore_errors=True); continue
                if waited % 60 == 0:
                    print(f'[{self.who}] waiting for the heavy-job lock ...', file=sys.stderr)
                time.sleep(5); waited += 5

    def __exit__(self, *a):
        import shutil; shutil.rmtree(self.LOCK, ignore_errors=True)
