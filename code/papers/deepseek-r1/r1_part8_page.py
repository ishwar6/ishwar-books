"""Build part8.md from results/part8_source.md, pasting code and log lines verbatim (so the page never drifts from
the scripts). Directives, each alone on a line:
  @@FILE path@@               the whole file, as a python block
  @@FUNC path name@@          one top-level function or constant of a file
  @@LINES path start end@@    lines start..end (1-based) of a file
  @@LOG name regex@@          every line of results/<name>_stdout.txt that matches regex, as a text block
  @@BLOCK name n regex@@      each line that matches regex plus the n lines after it, as one text block
  @@COUNT path@@ (inline)     the number of lines of a file"""
import ast, os, re

HERE = os.path.dirname(os.path.abspath(__file__))


def src(path):
    return open(os.path.join(HERE, path)).read()


def func(path, name):
    s = src(path); lines = s.split('\n')
    for node in ast.parse(s).body:
        names = [node.name] if hasattr(node, 'name') else [t.id for t in getattr(node, 'targets', []) if hasattr(t, 'id')]
        if name in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            return '\n'.join(lines[start - 1:node.end_lineno])
    raise SystemExit(f'{name} not in {path}')


def expand(line):
    m = re.fullmatch(r'@@(\w+) (.+)@@', line.strip())
    if not m:
        return line
    kind, arg = m.group(1), m.group(2)
    if kind == 'FILE':
        return '```python\n' + src(arg).rstrip('\n') + '\n```'
    if kind == 'FUNC':
        path, name = arg.split()
        return '```python\n' + func(path, name) + '\n```'
    if kind == 'LINES':
        path, a, b = arg.split()
        return '```python\n' + '\n'.join(src(path).split('\n')[int(a) - 1:int(b)]) + '\n```'
    if kind == 'BLOCK':
        name, n, rx = arg.split(' ', 2)
        lines = src(f'results/{name}_stdout.txt').split('\n')
        got = []
        for i, l in enumerate(lines):
            if re.search(rx, l):
                got += lines[i:i + 1 + int(n)]
        assert got, f'no log line for {arg}'
        return '```text\n' + '\n'.join(got) + '\n```'
    if kind == 'LOG':
        name, rx = arg.split(' ', 1)
        got = [l for l in src(f'results/{name}_stdout.txt').split('\n') if re.search(rx, l)]
        assert got, f'no log line for {arg}'
        return '```text\n' + '\n'.join(got) + '\n```'
    raise SystemExit(f'unknown directive {kind}')


text = re.sub(r'@@COUNT (\S+)@@', lambda m: str(len(src(m.group(1)).rstrip('\n').split('\n'))), src('results/part8_source.md'))
out = '\n'.join(expand(l) for l in text.split('\n'))
assert '@@' not in out, 'unexpanded directive: ' + out[out.index('@@') - 40: out.index('@@') + 60]
open(os.path.join(HERE, 'part8.md'), 'w').write(out)
print('wrote part8.md,', len(out.split()), 'words')
