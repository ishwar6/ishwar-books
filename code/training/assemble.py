"""Turn code/training/chN.md into a book chapter: replace {{FIG:key|caption}} with the SVG figures from results/figs_chN.json
and write content/books/how-models-are-trained/<part-folder>/<file>.md.  Usage: python assemble.py 1 [2 ...]
Refuses em dashes, unreplaced placeholders, missing images and a few words that must never appear."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
BOOK = os.path.join(ROOT, 'content', 'books', 'how-models-are-trained')
# chapter number -> (part folder, file name)
CHAPTERS = {
    '1': ('part1-the-big-picture', '01-from-next-word-to-assistant.md'),
    '2': ('part1-the-big-picture', '02-how-it-started.md'),
    '3': ('part1-the-big-picture', '03-how-we-measure-a-model.md'),
    '4': ('part2-base-to-instruction-follower', '04-pretraining.md'),
    '5': ('part2-base-to-instruction-follower', '05-supervised-fine-tuning.md'),
    '6': ('part3-learning-from-feedback', '06-reinforcement-learning-basics.md'),
    '7': ('part3-learning-from-feedback', '07-reward-models.md'),
    '8': ('part3-learning-from-feedback', '08-rlhf-with-ppo.md'),
}
EM_DASH = chr(0x2014)
BANNED = ['Alammar', 'YouTube']


def build(ch):
    src = os.path.join(HERE, f'ch{ch}.md')
    figs = os.path.join(HERE, 'results', f'figs_ch{ch}.json')
    F = json.load(open(figs)) if os.path.exists(figs) else {}
    s = open(src).read()
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
        print(f'ch {ch}: WARNING unused figures: {", ".join(unused)}')
    for img in re.findall(r'\]\((/img/[^)]+)\)|src="(/img/[^"]+)"', s):
        p = next(x for x in img if x)
        assert os.path.exists(os.path.join(ROOT, 'public', p.lstrip('/'))), f'missing image {p}'
    folder, name = CHAPTERS[ch]
    os.makedirs(os.path.join(BOOK, folder), exist_ok=True)
    open(os.path.join(BOOK, folder, name), 'w').write(s)
    print(f'ch {ch}: wrote {folder}/{name}, {len(s.split())} words, {len(used)} figures')


if __name__ == '__main__':
    for c in sys.argv[1:] or [c for c in CHAPTERS if os.path.exists(os.path.join(HERE, f'ch{c}.md'))]:
        build(c)
