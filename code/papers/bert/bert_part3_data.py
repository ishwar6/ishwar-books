"""Real text for the Part 3 experiments: WikiText-103 (Merity et al., 2016), a public set of Wikipedia articles.
WikiText stores text pre-split (" , " and " @-@ "), so we join it back into normal text first."""
import re
from datasets import load_dataset


def detok(s):
    s = s.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
    s = re.sub(r' ([.,;:!?%)\]}])', r'\1', s)
    s = re.sub(r'([(\[{$]) ', r'\1', s)
    s = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r'\1', s)
    s = re.sub(r'" (.*?) "', r'"\1"', s)
    return s.strip()


def articles(split='test', config='wikitext-103-raw-v1'):
    """List of articles; each article is a list of paragraphs (headings dropped)."""
    out, cur = [], None
    for line in load_dataset('Salesforce/wikitext', config)[split]['text']:
        if re.match(r'^ = [^=].* = $', line.rstrip('\n')):          # " = Title = " starts a new article
            cur = []
            out.append(cur)
        elif line.strip() and not line.strip().startswith('='):     # skip " = = Section = = " headings
            if cur is not None:
                cur.append(detok(line))
    return [a for a in out if a]


def sentences(paragraph):
    """A simple sentence splitter: good enough for building sentence pairs."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"])', paragraph) if len(s.split()) >= 4]
