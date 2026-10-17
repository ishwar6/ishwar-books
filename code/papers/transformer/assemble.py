"""Turn code/papers/transformer/partN.md into a site page: replace {{FIG:key|caption}} with the generated SVG figures
and write content/papers/transformer/<slug>.md.  Usage: python assemble.py 1 [2 ...]   (no argument: every part that exists)
Also refuses em dashes, unreplaced placeholders, missing images and a few words that must never appear,
and runs an ASD-STE100 (Simplified Technical English) check on our own prose (see ste_report).
Add --strict to refuse to write a part that has STE problems."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
OUT = os.path.join(ROOT, 'content', 'papers', 'transformer')
SLUGS = {
    'b': 'background-the-road-to-the-transformer',
    '1': 'part-1-the-big-idea',
    '2': 'part-2-the-architecture',
    '3': 'part-3-attention',
    '4': 'part-4-why-self-attention',
    '5': 'part-5-training',
    '6': 'part-6-results-and-impact',
    '7': 'part-7-build-it-from-scratch',
}
EM_DASH = chr(0x2014)   # never allowed in the site's text
BANNED = ['Alammar', 'ChatGPT', 'Miro ', 'lecture', 'YouTube', 'Vizuara', 'Claude']

# ASD-STE100 limits: 20 words for a procedural sentence, 25 for a descriptive one; 6 sentences in a paragraph.
MAX_WORDS, MAX_SENTENCES = 25, 6
# Common words that STE does not approve, and the approved word to use.
STE_WORDS = {
    'utilize': 'use', 'utilise': 'use', 'in order to': 'to', 'prior to': 'before', 'approximately': 'about',
    'numerous': 'many', 'commence': 'start', 'subsequently': 'then', 'facilitate': 'help',
    'sufficient': 'enough', 'obtain': 'get', 'demonstrate': 'show', 'additional': 'more',
    'whilst': 'while', 'hence': 'thus / so', 'ensure': 'make sure', 'in addition': 'also',
    'it is possible to': 'you can', 'a number of': 'some / many', 'due to': 'because of',
}


def prose_blocks(md):
    """Return (line_no, paragraph) for our own prose: no code, maths, tables, HTML, headings or paper boxes.
    Each list item is its own paragraph."""
    out, buf, start = [], [], 0
    in_code = in_math = in_paper = False

    def flush():
        if buf:
            out.append((start, ' '.join(buf)))
        buf.clear()

    for i, l in enumerate(md.split('\n')):
        s = l.strip()
        if s == '## References':                              # citations are data, not prose
            break
        if s.startswith('```'):
            in_code = not in_code; flush(); continue
        if s == '$$':
            in_math = not in_math; flush(); continue
        if in_code or in_math:
            continue
        if s.startswith('> [!PAPER]'):
            in_paper = True                                    # the screenshot and the quotes are the paper's words...
        elif in_paper and not s.startswith('>'):
            in_paper = False
        elif in_paper and s.startswith('> **What it means:**'):
            in_paper = False                                   # ...but "What it means" is our own prose
            flush(); continue
        if not s or in_paper or s.startswith(('#', '|', '<', '{{', '---', '> [!')) or s in ('>',):
            flush(); continue
        text = re.sub(r'^>\s?', '', s)
        if re.match(r'^([-*]|\d+\.)\s+', text):
            flush()
            text = re.sub(r'^([-*]|\d+\.)\s+', '', text)
        if not buf:
            start = i + 1
        buf.append(text)
    flush()
    return out


def sentences(p):
    p = re.sub(r'\$\$.*?\$\$', 'X', p)                        # inline maths counts as one word
    p = re.sub(r'`[^`]*`', 'X', p)
    p = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', p)             # links: keep the text
    p = re.sub(r'"[^"]*"', 'Q', p)                             # a quotation of the paper counts as one word
    p = re.sub(r'\*\*([^*]+)\*\*', r'\1', p)
    p = re.sub(r'\b(e\.g|i\.e|et al|vs|Fig|Eq|Sec|approx)\.', r'\1', p)
    p = re.sub(r'(\d)\.(\d)', r'\1\2', p)
    return [s.strip() for s in re.split(r'(?<=[.!?;])\s+(?=[A-Z0-9(])', p) if s.strip()]


def ste_report(md, part):
    long_s, long_p, words = [], [], []
    for ln, p in prose_blocks(md):
        ss = sentences(p)
        if len(ss) > MAX_SENTENCES:
            long_p.append((ln, len(ss)))
        for s in ss:
            n = len(re.findall(r"[A-Za-z0-9][\w'’×-]*", s))
            if n > MAX_WORDS:
                long_s.append((ln, n, s[:120]))
        low = ' ' + re.sub(r'"[^"]*"', 'Q', p).lower() + ' '   # quotations of the paper keep the paper's words
        for w, use in STE_WORDS.items():
            if re.search(r'\b' + re.escape(w) + r'\b', low):
                words.append((ln, w, use))
    print(f'part {part}: STE check: {len(long_s)} sentences over {MAX_WORDS} words, '
          f'{len(long_p)} paragraphs over {MAX_SENTENCES} sentences, {len(words)} unapproved words')
    for ln, n, s in long_s:
        print(f'   line {ln}: {n} words: {s}')
    for ln, n in long_p:
        print(f'   line {ln}: paragraph of {n} sentences')
    for ln, w, use in words:
        print(f'   line {ln}: "{w}" -> {use}')
    return len(long_s) + len(long_p) + len(words)


def build(part, strict=False):
    src = os.path.join(HERE, f'part{part}.md')
    figs = os.path.join(HERE, 'results', f'figs_part{part}.json')
    F = json.load(open(figs)) if os.path.exists(figs) else {}
    s = open(src).read()
    problems = ste_report(s, part)
    used = set()

    def rep(m):
        used.add(m.group(1))
        return f'<figure class="fig">{F[m.group(1)]}<figcaption>{m.group(2)}</figcaption></figure>'
    s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', rep, s)
    assert '{{' not in s, 'unreplaced figure placeholder'
    assert EM_DASH not in s, 'em dash found'
    for w in BANNED:
        assert w not in s, f'banned word: {w!r}'
    unused = sorted(set(F) - used)
    if unused:
        print(f'part {part}: WARNING unused figures: {", ".join(unused)}')
    for img in re.findall(r'\]\((/img/[^)]+)\)|src="(/img/[^"]+)"', s):
        p = next(x for x in img if x)
        assert os.path.exists(os.path.join(ROOT, 'public', p.lstrip('/'))), f'missing image {p}'
    if strict and problems:
        raise SystemExit(f'part {part}: fix the STE problems first')
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, SLUGS[part] + '.md'), 'w').write(s)
    print(f'part {part}: wrote {SLUGS[part]}.md, {len(s.split())} words, {len(used)} figures')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    parts = args or [p for p in SLUGS if os.path.exists(os.path.join(HERE, f'part{p}.md'))]
    for p in parts:
        build(p, strict='--strict' in sys.argv)
