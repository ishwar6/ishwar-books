"""Figures for Part 6, drawn from the saved runs in scratch/runs/ -> results/figs_part6.json.
  p6_map       the loop of Figure 2 with the names of our files
  p6_progress  best search accuracy so far, after each evaluated candidate, for both proposers
  p6_pareto    search accuracy vs prompt size for every candidate of the scripted run, with the Pareto frontier
  p6_gap       search accuracy vs test accuracy for every candidate of the scripted run (like the paper's Figure 7)"""
import json, os
from figlib import svg, text, box, arrow, esc

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, 'scratch', 'runs')
F = {}


def load(run):
    d = os.path.join(RUNS, run)
    return json.load(open(os.path.join(d, 'summary.json'))), json.load(open(os.path.join(d, 'run.json')))


# ------------------------------------------------------------------------------------------------ map
def fig_map():
    b = []
    W, H = 820, 330

    def node(x, y, w, h, title, sub, cls='box'):
        b.extend([box(x, y, w, h, cls), text(x + w / 2, y + 24, title, 't-title', 'middle')])
        for i, s in enumerate(sub):
            b.append(text(x + w / 2, y + 44 + 17 * i, s, 't-tick', 'middle'))
    node(20, 20, 220, 104, 'Filesystem D', ['runs/<run>/candidates/', 'harness.py, scores.json,', 'traces.jsonl, proposal.md'], 'box-1')
    node(300, 20, 220, 104, 'Proposer P', ['proposers.py: scripted rules,', 'or Qwen2.5-1.5B writing code', '(an API or agent: not run)'], 'box-2')
    node(580, 20, 220, 104, 'New harness H', ['one .py file:', 'class Harness', 'update(), predict()'])
    node(580, 190, 220, 104, 'validate()', ['harness_api.py', 'import, run on a fake model,', 'reject broken files'])
    node(300, 190, 220, 104, 'evaluate()', ['evaluate.py, frozen model M', 'Qwen2.5-0.5B-Instruct', 'memory 60, search set 50'], 'box-3')
    b += [arrow(240, 60, 296, 60, True), text(268, 50, 'reads', 't-tick', 'middle'),
          arrow(520, 60, 576, 60, True), text(548, 50, 'writes', 't-tick', 'middle'),
          arrow(690, 124, 690, 186, True),
          arrow(580, 242, 524, 242, True), text(552, 232, 'passed', 't-tick', 'middle'),
          arrow(300, 242, 130, 242), arrow(130, 242, 130, 128),
          text(215, 232, 'new folder', 't-tick', 'middle')]
    b += [box(20, 330, 500, 50, 'box-ghost'), text(270, 352, 'heldout/: the test set (100 messages)', 't-tick', 'middle'),
          text(270, 370, 'scored once, after the search; no proposer reads it', 't-muted', 'middle')]
    F['p6_map'] = svg(W, H + 60, 'The search loop of the paper, with the files of our package', b)


# ------------------------------------------------------------------------------------------------ charts
def axes(L, T, pw, ph, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, fmtx=str, fmty=str):
    X = lambda v: L + (v - xmin) / (xmax - xmin) * pw
    Y = lambda v: T + ph - (v - ymin) / (ymax - ymin) * ph
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, fmty(t), 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), T + ph + 20, fmtx(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    b.append(text(L + pw / 2, T + ph + 44, xlabel, 't-tick', 'middle'))
    b.append(text(L - 50, T - 12, ylabel, 't-tick'))
    return X, Y, b


def fig_progress(runs):
    W, H, L, T = 820, 360, 64, 34
    pw, ph = 560, 260
    n = max(len(s) for s, _ in runs.values())
    allv = [r['search_acc'] for s, _ in runs.values() for r in s]
    ymin, ymax = 10 * (min(allv) // 10), 10 * (max(allv) // 10 + 1)
    X, Y, b = axes(L, T, pw, ph, 0.5, n + 0.5, ymin, ymax, range(1, n + 1), range(int(ymin), int(ymax) + 1, 10),
                   'evaluated candidate (the first three are the seeds)', 'search accuracy (%)')
    ends = []
    for (name, (summ, meta)), (lc, dc) in zip(runs.items(), [('l1', 's1'), ('l2', 's2')]):
        best, pts = -1, []
        for i, r in enumerate(summ, 1):
            best = max(best, r['search_acc'])
            pts.append((i, best))
            b.append(f'<g class="mark"><title>{esc(name)}: {esc(r["folder"])}, {r["search_acc"]}%</title>'
                     f'<circle class="{dc}" cx="{X(i):.1f}" cy="{Y(r["search_acc"]):.1f}" r="4" fill-opacity="0.45"/></g>')
        path = ' '.join(f'{X(x):.1f},{Y(v):.1f}' for x, v in pts)
        b.append(f'<polyline class="{lc}" points="{path}"/>')
        ends.append((Y(pts[-1][1]), f'{name}: best {pts[-1][1]}%', dc, X(pts[-1][0])))
    ends.sort()
    for i in range(1, len(ends)):
        if ends[i][0] < ends[i - 1][0] + 18:
            ends[i] = (ends[i - 1][0] + 18, *ends[i][1:])
    for y, lab, dc, x in ends:
        b += [f'<rect class="{dc}" x="{L + pw + 14}" y="{y - 5:.1f}" width="10" height="10" rx="2"/>',
              text(L + pw + 30, y + 4, lab, 't-note')]
    b.append(text(L + pw + 14, T + ph - 6, 'dots: each candidate', 't-muted'))
    b.append(text(L + pw + 14, T + ph + 12, 'line: best so far', 't-muted'))
    F['p6_progress'] = svg(W, H + 20, 'Best search accuracy so far after each evaluated candidate, for the two proposers', b)


def fig_pareto(summ):
    W, H, L, T = 820, 360, 64, 34
    pw, ph = 600, 260
    xs = [r['ctx_chars'] for r in summ]
    ys = [r['search_acc'] for r in summ]
    xmax = 1000 * (max(xs) // 1000 + 1)
    ymin, ymax = 10 * (min(ys) // 10), 10 * (max(ys) // 10 + 1)
    X, Y, b = axes(L, T, pw, ph, 0, xmax, ymin, ymax, range(0, xmax + 1, 1000), range(int(ymin), int(ymax) + 1, 10),
                   'prompt characters per message (lower is cheaper)', 'search accuracy (%)', fmtx=lambda v: f'{v:,}')
    front = sorted([r for r in summ if r['frontier']], key=lambda r: r['ctx_chars'])
    step = []
    for r in front:
        if step:
            step.append((r['ctx_chars'], step[-1][1]))
        step.append((r['ctx_chars'], r['search_acc']))
    step.append((xmax, front[-1]['search_acc']))
    b.append('<polyline class="l2" points="' + ' '.join(f'{X(x):.1f},{Y(y):.1f}' for x, y in step) + '"/>')
    for r in summ:
        cls = 's2' if r['frontier'] else 's1'
        b.append(f'<g class="mark"><title>{esc(r["folder"])}: {r["search_acc"]}%, {r["ctx_chars"]:,} characters</title>'
                 f'<circle class="{cls} ring" cx="{X(r["ctx_chars"]):.1f}" cy="{Y(r["search_acc"]):.1f}" r="{6 if r["frontier"] else 4.5}"/></g>')
    groups = {}
    for r in summ:                                      # candidates with the same two numbers share one label
        if r['frontier'] or r['seed']:
            groups.setdefault((r['ctx_chars'], r['search_acc']), []).append(r['folder'])
    for (cx, acc), folders in groups.items():
        lab = folders[0][4:].replace('_', ' ') if len(folders) == 1 and folders[0][4:] in ('zero_shot', 'few_shot_8', 'few_shot_all') else ' and '.join(f[:3] for f in folders)
        b.append(text(X(cx) + 9, Y(acc) - 9, lab, 't-tick'))
    F['p6_pareto'] = svg(W, H + 20, 'Search accuracy against prompt size for every candidate, with the Pareto frontier', b)


def fig_gap(summ):
    W, H, L, T = 820, 400, 64, 34
    pw, ph = 380, 300
    rows = [r for r in summ if 'test_acc' in r]
    lo = 10 * (min(min(r['search_acc'], r['test_acc']) for r in rows) // 10)
    hi = 10 * (max(max(r['search_acc'], r['test_acc']) for r in rows) // 10 + 1)
    X, Y, b = axes(L, T, pw, ph, lo, hi, lo, hi, range(int(lo), int(hi) + 1, 10), range(int(lo), int(hi) + 1, 10),
                   'search accuracy (%)', 'test accuracy (%)')
    b.append(f'<line class="base-line" x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}"/>')
    b.append(text(X(hi) - 4, Y(hi - 4) + 4, 'y = x', 't-muted', 'start'))
    for r in rows:
        cls = 's3' if r['seed'] else ('s2' if r['frontier'] else 's1')
        b.append(f'<g class="mark"><title>{esc(r["folder"])}: search {r["search_acc"]}%, test {r["test_acc"]}%</title>'
                 f'<circle class="{cls} ring" cx="{X(r["search_acc"]):.1f}" cy="{Y(r["test_acc"]):.1f}" r="5.5"/></g>')
    lx = L + pw + 40
    for i, (cls, lab) in enumerate([('s3', 'seed'), ('s2', 'proposed, on the search frontier'), ('s1', 'proposed, other')]):
        b += [f'<circle class="{cls}" cx="{lx + 5}" cy="{T + 20 + 24 * i}" r="5.5"/>', text(lx + 18, T + 25 + 24 * i, lab, 't-note')]
    b.append(text(lx, T + 110, 'Above the line: better on the test set', 't-muted'))
    b.append(text(lx, T + 128, 'than on the search set. Below: worse.', 't-muted'))
    F['p6_gap'] = svg(W, H + 10, 'Search accuracy against test accuracy for every candidate of the scripted run', b)


if __name__ == '__main__':
    fig_map()
    runs = {}
    for name, label in [('scripted-1', 'scripted'), ('local-llm-1', 'local LLM')]:
        if os.path.exists(os.path.join(RUNS, name, 'summary.json')):
            runs[label] = load(name)
    if runs:
        fig_progress(runs)
    if 'scripted' in runs:
        fig_pareto(runs['scripted'][0])
        fig_gap(runs['scripted'][0])
    json.dump(F, open(os.path.join(HERE, 'results', 'figs_part6.json'), 'w'))
    print('figures:', ', '.join(F))
