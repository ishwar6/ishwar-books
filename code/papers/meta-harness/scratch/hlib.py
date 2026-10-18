"""Small helpers that any harness may import. The proposer may use them but must not edit this file.

  words(text)            lower-case word tokens
  TfIdf                  TF-IDF retrieval over the harness's memory, written from scratch (no sklearn)
  parse_label(reply, labels)  turn the model's free-text reply into one valid label (or None)
  show_label(label)      'card_arrival' -> 'card arrival' (some prompts read better without underscores)
"""
import difflib, math, re
from collections import Counter

WORD = re.compile(r"[a-z0-9']+")


def words(text):
    return WORD.findall(text.lower())


class TfIdf:
    """TF-IDF vectors with cosine similarity.

    tf(w, d)  = how many times word w occurs in document d
    idf(w)    = log((1 + N) / (1 + df(w))) + 1, where N = number of documents, df(w) = documents that contain w
    vector(d) = tf(w, d) * idf(w) for each word, then divided by its length (L2 norm), so cosine = dot product.
    These are the same formulas as sklearn's TfidfVectorizer with its default settings (smooth_idf=True, norm='l2').
    """

    def __init__(self, docs):
        self.docs = list(docs)
        self.n = len(self.docs)
        df = Counter(w for d in self.docs for w in set(words(d)))
        self.idf = {w: math.log((1 + self.n) / (1 + c)) + 1 for w, c in df.items()}
        self.vecs = [self.vector(d) for d in self.docs]

    def vector(self, text):
        tf = Counter(w for w in words(text) if w in self.idf)        # words never seen in memory are dropped
        v = {w: c * self.idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def scores(self, text):
        q = self.vector(text)
        return [sum(q[w] * dv.get(w, 0.0) for w in q) for dv in self.vecs]   # cosine with every document

    def top(self, text, k):
        """Indices of the k most similar documents, best first."""
        s = self.scores(text)
        return sorted(range(self.n), key=lambda i: -s[i])[:k]


def show_label(label):
    return label.replace('_', ' ')


def parse_label(reply, labels):
    """Map a reply such as 'Label: card_arrival.' or 'card arrival' to a label; None if nothing fits."""
    r = reply.lower().replace(' ', '_')
    hits = [l for l in labels if l in r]
    if hits:
        return max(hits, key=len)                              # the longest label that appears in the reply
    first = r.strip().split('\n')[0].strip(' .:"\'`*')
    close = difflib.get_close_matches(first, labels, n=1, cutoff=0.6)
    return close[0] if close else None
