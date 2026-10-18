"""Part 4: the math retrieval harness of §4.2 and Appendix B.2, rebuilt on a small public math set (no model needed).

What it shows (every printed line is saved to results/part4_stdout.txt):
  1. BM25 by hand on five tiny documents: idf, term frequency, length norm and the final score.
  2. Method 1 (our BM25, from scratch) against Method 2 (the rank_bm25 library) on the full corpus: same scores.
  3. A naive tokenizer against a math-aware one that keeps LaTeX pieces such as \\frac and ^{2} whole.
  4. Decontamination as in Appendix C.2: exact prefix match and word Jaccard >= 0.8.
  5. The lexical router of Figure 8: keyword sets and regex cues, scored against the dataset's own subject labels.
  6. The four route policies (BM25@20 + dedup + rerank, 1 hard ref + 2 BM25, BM25@12 + technique bonus, adaptive K).
  7. How often the retrieved examples share the query's subject: plain BM25 top-3 against the routed harness.

Data: MATH (Hendrycks et al. 2021) from the Hugging Face hub (EleutherAI/hendrycks_math). The 7,500 training problems
are the retrieval corpus; test problems are the queries. This is far smaller and easier than the paper's 535K-problem
corpus and its IMO-level test set. rank_bm25 is installed outside the venv: PYTHONPATH=<scratchpad>/mhlib-p4."""
import math, re, random, time
from collections import Counter, defaultdict
from itertools import chain
from common import Log, save

SUBJECTS = ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory',
            'prealgebra', 'precalculus']
# The paper's four routes. MATH has seven subjects; four of them have no route of their own and go to "default".
ROUTE_OF_SUBJECT = {'Counting & Probability': 'combinatorics', 'Geometry': 'geometry',
                    'Number Theory': 'number_theory'}


def load_math():
    from datasets import load_dataset
    train, test = [], []
    for s in SUBJECTS:
        d = load_dataset('EleutherAI/hendrycks_math', s)
        train += list(d['train'])
        test += list(d['test'])
    for row in train + test:
        lv = row['level'].replace('Level ', '')
        row['difficulty'] = int(lv) if lv.isdigit() else 3       # a few rows say "Level ?"
        row['route_true'] = ROUTE_OF_SUBJECT.get(row['type'], 'default')
    return train, test


# ---------------------------------------------------------------- tokenizers
def tokenize_naive(text):
    """Lower-case words and numbers. LaTeX falls apart: \\frac{1}{2} becomes frac, 1, 2."""
    return re.findall(r'\w+', text.lower())


MATH_TOKEN = re.compile(r'\\[A-Za-z]+'          # a LaTeX command: \frac, \sqrt, \angle
                        r'|[\^_]\{[^{}]*\}'      # a superscript or subscript group: ^{2}, _{10}
                        r'|[\^_]\w'              # a short one: ^2, _7
                        r'|\d+'                  # a number
                        r'|[A-Za-z]+')           # a word


def tokenize_math(text):
    """Math-aware tokens: every LaTeX command and every ^{...} or _{...} group stays one token."""
    return [t if t.startswith(('\\', '^', '_')) else t.lower() for t in MATH_TOKEN.findall(text)]


# ---------------------------------------------------------------- Method 1: BM25 from scratch
def bm25_index(docs, k1=1.5, b=0.75, eps=0.25):
    """docs: a list of token lists. Builds what BM25 needs: idf per word, and postings (doc id, count) per word."""
    N = len(docs)
    dl = [len(d) for d in docs]                       # document length in tokens
    avgdl = sum(dl) / N                               # average document length
    postings = defaultdict(list)                      # word -> [(doc id, term frequency in that doc)]
    for i, d in enumerate(docs):
        for w, f in Counter(d).items():
            postings[w].append((i, f))
    idf = {w: math.log((N - len(p) + 0.5) / (len(p) + 0.5)) for w, p in postings.items()}
    floor = eps * sum(idf.values()) / len(idf)        # words in more than half the docs get a small positive idf
    idf = {w: (v if v >= 0 else floor) for w, v in idf.items()}
    return dict(N=N, dl=dl, avgdl=avgdl, postings=postings, idf=idf, k1=k1, b=b)


def bm25_scores(ix, query):
    """BM25 score of every document for a query (a token list). Only docs that contain a query word get work."""
    k1, b = ix['k1'], ix['b']
    scores = [0.0] * ix['N']
    for q in query:                                   # a word repeated in the query counts each time
        for d, f in ix['postings'].get(q, []):
            norm = k1 * (1 - b + b * ix['dl'][d] / ix['avgdl'])     # long documents need more hits
            scores[d] += ix['idf'][q] * f * (k1 + 1) / (f + norm)  # saturates: the 10th hit adds little
    return scores


def top_k(scores, k):
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]


# ---------------------------------------------------------------- decontamination (Appendix C.2)
def words(text):
    return set(re.findall(r'\w+', text.lower()))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def contaminated(query, corpus_words, word_postings, corpus_prefixes, n_prefix=60, thresh=0.8):
    """True if a corpus problem starts with the same 60 characters, or shares >= 80% of its word set.
    word_postings maps a word to the corpus problems that contain it, so we count shared words in one pass."""
    if query[:n_prefix].strip().lower() in corpus_prefixes:
        return True
    q = words(query)
    shared = Counter(chain.from_iterable(word_postings.get(w, ()) for w in q))   # doc id -> |q & doc|
    return any(n / (len(q) + len(corpus_words[d]) - n) >= thresh for d, n in shared.items())


# ---------------------------------------------------------------- the lexical router (Figure 8)
KEYWORDS = {
    'combinatorics': ['how many ways', 'probability', 'arrange', 'arrangements', 'committee', 'choose', 'chosen',
                      'dice', 'die is', 'coin', 'cards', 'deck', 'permutation', 'randomly', 'at random',
                      'expected value', 'distinguishable', 'binom', 'outcomes', 'subsets', 'how many different'],
    'geometry': ['triangle', 'circle', 'angle', 'radius', 'diameter', 'polygon', 'perimeter', 'hexagon',
                 'rectangle', 'trapezoid', 'parallelogram', 'sphere', 'cone', 'cylinder', 'cube', 'prism',
                 'perpendicular', 'parallel', 'inscribed', 'circumference', 'hypotenuse', 'isosceles', 'pyramid'],
    'number_theory': ['remainder', 'divisible', 'prime', 'divisor', 'divisors', 'factors', 'gcd', 'lcm',
                      'greatest common', 'least common multiple', 'modulo', 'pmod', 'units digit', 'digit',
                      'base', 'congruent', 'multiple of', 'integer n'],
}
GEOMETRY_REGEX = [r'\\angle', r'\\triangle', r'\\overline\{[A-Z]{2}\}', r'\\pi r', r'\b[A-Z]{3}\b',
                  r'\b[A-Z]{2}\s*=\s*\d']        # notation such as triangle ABC, AB = 5


def route(problem):
    """Count the cues of each route; the route with the most cues wins. No cue at all: the default route."""
    text = problem.lower()
    hits = {r: sum(text.count(k) for k in ks) for r, ks in KEYWORDS.items()}
    hits['geometry'] += sum(len(re.findall(p, problem)) for p in GEOMETRY_REGEX)
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else 'default'


# ---------------------------------------------------------------- the four route policies (Appendix B.2)
TECHNIQUE = ['modulo', 'mod ', 'prime factorization', 'euclidean algorithm', 'chinese remainder', 'fermat',
             'divisibility rule', 'congruen', 'factor', 'parity']


def dedup(cands, corpus, keep, thresh=0.8):
    """Walk down the ranked list; drop a candidate whose words overlap >= 80% with one already kept."""
    out = []
    for i in cands:
        wi = words(corpus[i]['problem'])
        if all(jaccard(wi, words(corpus[j]['problem'])) < thresh for j in out):
            out.append(i)
        if len(out) == keep:
            break
    return out


def rerank(cands, scores, corpus, w_diff=0.15, w_tech=0.0):
    """New score = BM25 score / best BM25 score + w_diff * difficulty/5 (+ w_tech if the solution names a technique
    in its first 200 characters). The paper does not give its weights; these are ours."""
    top = max(scores[i] for i in cands) or 1.0

    def s(i):
        tech = any(t in corpus[i]['solution'][:200].lower() for t in TECHNIQUE)
        return scores[i] / top + w_diff * corpus[i]['difficulty'] / 5 + (w_tech if tech else 0.0)
    return sorted(cands, key=lambda i: -s(i))


def adaptive_k(cands, scores):
    """Default route: one example if the top hit stands far above the rest, else two or three."""
    s = [scores[i] for i in cands]
    if s[0] >= 1.5 * s[1]:
        return 1
    return 2 if s[0] >= 1.2 * s[2] else 3


def retrieve(problem, ix, corpus, hard_geometry, tokenize=tokenize_math):
    """The routed harness: pick a route, then run that route's retrieval policy. Returns (route, doc ids)."""
    r = route(problem)
    scores = bm25_scores(ix, tokenize(problem))
    if r == 'combinatorics':                                      # BM25@20, dedup to 8, rerank, keep 3
        cands = dedup(top_k(scores, 20), corpus, keep=8)
        return r, rerank(cands, scores, corpus)[:3]
    if r == 'geometry':                                           # 1 hard reference + 2 raw BM25 neighbours
        hard = max(hard_geometry, key=lambda i: scores[i])
        return r, [hard] + [i for i in top_k(scores, 3) if i != hard][:2]
    if r == 'number_theory':                                      # BM25@12, rerank with a technique bonus, keep 3
        return r, rerank(top_k(scores, 12), scores, corpus, w_tech=0.1)[:3]
    cands = rerank(top_k(scores, 10), scores, corpus)             # default: BM25@10, rerank, adaptive K
    return r, cands[:adaptive_k(cands, scores)]


def main():
    log = Log('part4')
    R = {}
    log('== 1. BM25 by hand: five tiny documents, query "prime remainder"')
    docs = ['find the remainder when 100 is divided by 7',
            'the prime 7 and the prime 11 and the prime 13',
            'the radius of the circle is 6',
            'is 91 a prime',
            'the area of the square is 36']
    toks = [d.split() for d in docs]
    ix = bm25_index(toks)
    log(f'N = {ix["N"]}, lengths = {ix["dl"]}, avgdl = {ix["avgdl"]:.3f}')
    for w in ['prime', 'remainder', 'the']:
        log(f'idf({w}) = {ix["idf"][w]:.4f}   (in {len(ix["postings"][w])} of 5 docs)')
    for d in [1, 3]:                                  # doc 2 says "prime" 3 times; doc 4 once, and is short
        f = toks[d].count('prime')
        norm = 1.5 * (1 - 0.75 + 0.75 * ix['dl'][d] / ix['avgdl'])
        log(f'doc {d + 1}: tf(prime) = {f}, length {ix["dl"][d]}, k1 * (1 - b + b * dl/avgdl) = {norm:.4f}, '
            f'tf part = {f * 2.5 / (f + norm):.4f}')
    sat = {}
    for ratio in [0.5, 1.0, 2.0]:                     # a short, an average and a long document
        norm = 1.5 * (1 - 0.75 + 0.75 * ratio)
        sat[ratio] = [f * 2.5 / (f + norm) for f in range(1, 11)]
        log(f'tf part, dl = {ratio} x avgdl, tf = 1..10:', [round(v, 2) for v in sat[ratio]])
    s = bm25_scores(ix, ['prime', 'remainder'])
    log('scores:', [round(v, 4) for v in s])
    from rank_bm25 import BM25Okapi
    lib = BM25Okapi(toks)
    s2 = lib.get_scores(['prime', 'remainder'])
    log('rank_bm25:', [round(float(v), 4) for v in s2], f'max diff {max(abs(a - b) for a, b in zip(s, s2)):.1e}')
    R['tiny'] = dict(sat={str(k): v for k, v in sat.items()}, scores=s, lib=list(map(float, s2)), dl=ix['dl'], avgdl=ix['avgdl'],
                     idf={w: ix['idf'][w] for w in ['prime', 'remainder', 'the', 'circle']})

    log('')
    log('== 2. Tokenizers on one problem')
    ex = r'Compute $\frac{1}{2}+x^{2}$ where $x^2 = 9$ and $\angle ABC = 90^\circ$.'
    log('text:', ex)
    log('naive:', tokenize_naive(ex))
    log('math :', tokenize_math(ex))

    log('')
    log('== 3. The corpus: MATH train = corpus, MATH test = queries')
    t = time.time()
    train, test = load_math()
    log(f'corpus {len(train)} problems, test {len(test)} problems ({time.time() - t:.1f} s to load)')
    log('test subjects:', dict(Counter(r['type'] for r in test)))
    t = time.time()
    ix = bm25_index([tokenize_math(r['problem']) for r in train])
    log(f'Method 1 (ours): indexed in {time.time() - t:.1f} s, {len(ix["idf"])} distinct tokens, '
        f'avgdl {ix["avgdl"]:.1f}')
    t = time.time()
    lib = BM25Okapi([tokenize_math(r['problem']) for r in train])
    log(f'Method 2 (rank_bm25): indexed in {time.time() - t:.1f} s')
    random.seed(0)
    qs = random.sample(test, 20)
    worst, same = 0.0, 0
    for q in qs:
        a = bm25_scores(ix, tokenize_math(q['problem']))
        b = lib.get_scores(tokenize_math(q['problem']))
        worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
        same += top_k(a, 5) == [int(i) for i in sorted(range(len(b)), key=lambda i: -b[i])[:5]]
    log(f'20 queries: largest score difference {worst:.1e}; same top-5 list for {same} of 20')
    R['agree'] = dict(worst=worst, same_top5=same)

    log('')
    log('== 4. Decontamination (Appendix C.2): prefix match or word Jaccard >= 0.8')
    cw = [words(r['problem']) for r in train]
    cp = {r['problem'][:60].strip().lower() for r in train}
    wp = defaultdict(list)
    for i, ws in enumerate(cw):
        for w in ws:
            wp[w].append(i)
    t = time.time()
    flagged = [q for q in test if contaminated(q['problem'], cw, wp, cp)]
    log(f'{len(flagged)} of {len(test)} test problems match a corpus problem ({time.time() - t:.0f} s)')
    for q in flagged[:2]:
        log('  e.g.', q['problem'][:90].replace('\n', ' '))
    clean = [q for q in test if q not in flagged]
    R['decon'] = dict(flagged=len(flagged), test=len(test))

    log('')
    log('== 5. The lexical router, scored against the dataset\'s subject labels')
    routes = ['combinatorics', 'geometry', 'number_theory', 'default']
    for name, rows in [('train', train), ('test', clean)]:
        conf = Counter((r['route_true'], route(r['problem'])) for r in rows)
        acc = sum(v for (a, b), v in conf.items() if a == b) / len(rows)
        log(f'{name}: {len(rows)} problems, router accuracy {100 * acc:.1f}%')
        if name == 'test':
            log(f'{"true route":>14} | ' + ' '.join(f'{r[:8]:>8}' for r in routes) + ' | correct')
            for a in routes:
                row = [conf[(a, b)] for b in routes]
                log(f'{a:>14} | ' + ' '.join(f'{v:8d}' for v in row) + f' | {100 * conf[(a, a)] / sum(row):5.1f}%')
            R['router'] = dict(acc=acc, conf={f'{a}|{b}': conf[(a, b)] for a in routes for b in routes})
        else:
            R['router_train_acc'] = acc
    for p in ['How many ways can 4 books be arranged on a shelf?',
              'What is the remainder when $2^{100}$ is divided by 7?',
              'In triangle $ABC$, $AB = 5$ and $\\angle C = 90^\\circ$. Find $BC$.',
              'Solve for $x$: $3x + 7 = 22$.']:
        log(f'  route({p!r}) = {route(p)}')

    log('')
    log('== 6. The routed harness on one test problem per route')
    hard_geo = [i for i, r in enumerate(train) if r['type'] == 'Geometry' and r['difficulty'] == 5]
    log(f'hard geometry index: {len(hard_geo)} level-5 geometry problems')
    random.seed(1)
    shown = set()
    for q in random.sample(clean, 200):
        r, ids = retrieve(q['problem'], ix, train, hard_geo)
        if r in shown or r != q['route_true']:
            continue
        shown.add(r)
        log(f'[{r}] query: {q["problem"][:80]!r}')
        for i in ids:
            log(f'    -> {train[i]["type"]}, level {train[i]["difficulty"]}: {train[i]["problem"][:60]!r}')
        if len(shown) == 4:
            break

    log('')
    log('== 7. Do the retrieved examples share the query\'s subject? (400 test problems)')
    random.seed(2)
    sample = random.sample(clean, 400)
    stats = {'bm25 top-3 (naive tokens)': [], 'bm25 top-3 (math tokens)': [], 'routed harness': []}
    nk = []
    ix_naive = bm25_index([tokenize_naive(r['problem']) for r in train])
    t = time.time()
    for q in sample:
        ids = top_k(bm25_scores(ix_naive, tokenize_naive(q['problem'])), 3)
        stats['bm25 top-3 (naive tokens)'].append(sum(train[i]['type'] == q['type'] for i in ids) / 3)
        ids = top_k(bm25_scores(ix, tokenize_math(q['problem'])), 3)
        stats['bm25 top-3 (math tokens)'].append(sum(train[i]['type'] == q['type'] for i in ids) / 3)
        r, ids = retrieve(q['problem'], ix, train, hard_geo)
        stats['routed harness'].append(sum(train[i]['type'] == q['type'] for i in ids) / len(ids))
        nk.append(len(ids))
    for k, v in stats.items():
        log(f'{k:>26}: {100 * sum(v) / len(v):.1f}% of retrieved examples share the subject')
    log(f'routed harness: {sum(nk) / len(nk):.2f} examples per problem on average '
        f'({time.time() - t:.0f} s for 400 queries x 3 methods)')
    R['subject_match'] = {k: sum(v) / len(v) for k, v in stats.items()}
    save('part4', R)


if __name__ == '__main__':
    main()
