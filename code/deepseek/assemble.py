"""Turn code/deepseek/partN.md into content/writings/<slug>.md.

    python assemble.py 3            # one part
    python assemble.py              # every part whose partN.md exists
    python assemble.py 3 --strict   # refuse to write if the ASD-STE100 check finds problems

Replaces {{FIG:key|caption}} with SVG from results/figs_partN.json. Refuses em dashes, banned words,
unresolved placeholders and missing images, and prints the STE report (sentences over 25 words,
paragraphs over 6 sentences, unapproved words) for our own prose.
"""
import importlib.util, json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLUGS = {
    '1': 'deepseek-1-every-model-and-what-is-open',
    '2': 'deepseek-2-the-v3-architecture',
    '3': 'deepseek-3-training-v3',
    '4': 'deepseek-4-r1-and-reasoning-by-rl',
    '5': 'deepseek-5-sparse-attention-and-v3-2',
    '6': 'deepseek-6-v4-and-v4-1',
    '7': 'deepseek-7-running-deepseek',
    '8': 'deepseek-8-the-specialised-models',
    '9': 'deepseek-9-deepinfer',
}
BANNED = ['Alammar', 'ChatGPT', 'Miro ', 'lecture', 'YouTube', 'Vizuara', 'Claude', 'plain English',
          'simple words', 'simply put', "Let's dive"]

spec = importlib.util.spec_from_file_location('ste', ROOT / 'code/papers/vllm/assemble.py')
ste = importlib.util.module_from_spec(spec); spec.loader.exec_module(ste)


def build(part, strict=False):
    src = (HERE / f'part{part}.md').read_text()
    figs_path = HERE / f'results/figs_part{part}.json'
    figs = json.loads(figs_path.read_text()) if figs_path.exists() else {}
    def fig(m):
        if m[1] not in figs:
            sys.exit(f'part {part}: figure {m[1]} missing from {figs_path.name}')
        return f'<figure class="fig">{figs[m[1]]}<figcaption>{m[2]}</figcaption></figure>'
    s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', fig, src)
    assert '{{' not in s, f'part {part}: unresolved placeholder near ' + re.search(r'.{0,40}\{\{.{0,40}', s)[0]
    assert chr(0x2014) not in s, f'part {part}: em dash found'
    for w in BANNED:
        assert w not in s, f'part {part}: banned phrase {w!r}'
    for img in re.findall(r'\]\((/img/[^)\s]+)\)', s):
        assert (ROOT / 'public' / img.lstrip('/')).is_file(), f'part {part}: missing image {img}'
    body = re.sub(r'<figure.*?</figure>', '', s.split('\n---\n', 1)[1], flags=re.S)
    body = re.sub(r'<details>.*?</details>', '', body, flags=re.S)
    problems = ste.ste_report(body, part)
    if strict and problems:
        sys.exit(f'part {part}: STE problems, not written')
    out = ROOT / 'content/writings' / f'{SLUGS[part]}.md'
    out.write_text(s)
    print(f'part {part}: wrote {out.name}, {len(body.split())} words of prose, {s.count("<figure")} figures')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    for p in args or [p for p in SLUGS if (HERE / f'part{p}.md').exists()]:
        build(p, strict='--strict' in sys.argv)
