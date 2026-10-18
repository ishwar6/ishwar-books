---
title: "Math Retrieval and TerminalBench-2"
description: "How Meta-Harness found a way to look up solved problems for olympiad math, and a better harness for an agent that works in a terminal. We build the search engine and the discovered retriever in Python, test them with a small local model, and run two key pieces of the released agent code."
part: 4
covers: "§4.2, §4.3, Tables 6, 7, 10, 11, Figures 8, 9, Appendices A.2, B.2, B.3, C.2"
date: 2026-10-18
tags: [meta-harness, retrieval, bm25, agents, terminal-bench]
---

The same search loop that wrote text classifiers in [Part 3](part-3-text-classification.md) gets two harder jobs here. In the first, it decides which solved problems a model sees before it attacks an olympiad problem. In the second, it changes the code around an agent that must finish long tasks in a terminal with no human help. Each piece below shows the exact lines of the paper, then explains them, and then rebuilds them in Python where a laptop can.

## Why look up solved problems for olympiad math {§4.2}

> [!PAPER] · §4.2
> [![The first lines of Section 4.2, with augmenting the model with the ability to retrieve examples from a large corpus and reusable proof patterns highlighted](/img/papers/meta-harness/p4-why-retrieval-1.png)](/img/papers/meta-harness/p4-why-retrieval-1.png)
> [![The next lines on the following page, with naive retrieval rarely surfaces the right traces in the right form, discovering the right retrieval policy, and allow the retrieval behavior itself to emerge from search highlighted](/img/papers/meta-harness/p4-why-retrieval-2.png)](/img/papers/meta-harness/p4-why-retrieval-2.png)
>
> **What it says:** the authors augment the model "with the ability to retrieve examples from a large corpus". Solutions often share "reusable proof patterns". But "naive retrieval rarely surfaces the right traces in the right form", so success depends on "discovering the right retrieval policy". They let that policy "emerge from search".
>
> **What it means:**
>
> A student who prepares for a math olympiad does not only learn facts. They solve hundreds of old problems, and they see the same tricks again and again. Count the complement. Look at the problem modulo 3. Draw the circle through three points. When a new problem looks like an old one, the old solution shows the way.
>
> The idea here is to give a model the same help. Before the model sees a new problem, a program searches a large collection of solved problems and puts a few of them into the prompt. The model can then copy a method, not an answer.
>
> The authors admit that this has rarely worked for hard math. A search that matches words can bring back a problem that looks similar but needs a different idea. Too many examples can also bury the real question. So the hard part is the **policy**: which problems to fetch, how many, and in what order. The paper does not design that policy by hand. It lets Meta-Harness write it.

> [!DEFINITION] Retrieval and corpus
> **Retrieval** means finding the items in a large collection that best match a query, the way a search engine finds web pages. The collection is the **corpus**. Here the query is a new math problem, and the corpus holds solved problems with their solutions.

> [!DEFINITION] Retrieval policy
> The rules a program follows to retrieve: which search method, how many results, how to sort them, and which ones to drop. In this section, the retrieval policy is the harness that Meta-Harness searches over.

## The corpus, and how it was cleaned {§4.2, Appendix C.2}

> [!PAPER] · §4.2
> [![The paragraph on the corpus, with 500,000 solved problems from eight open-source datasets and deduplicated and decontaminated highlighted](/img/papers/meta-harness/p4-corpus.png)](/img/papers/meta-harness/p4-corpus.png)
>
> **What it says:** the corpus "contains ≥500,000 solved problems from eight open-source datasets". The authors "deduplicated and decontaminated" it against the test problems and the search set, checked for exact prefix matches, and looked at the top BM25 results for held-out problems by hand.
>
> **What it means:**
>
> A corpus of solved problems has a trap. If a test problem, or a close copy of it, sits in the corpus, retrieval hands the model the answer. The score then measures copying, not reasoning. So the authors remove every corpus problem that matches a test problem, and every problem that is a repeat of another.
>
> They also open the search results and read them. That step catches what a filter misses, such as the same problem with new numbers.

> [!DEFINITION] Deduplication and decontamination
> **Deduplication** removes repeated items from a collection. **Decontamination** removes items from the training or retrieval data that also occur in the test data. Without it, a test score can be high only because the answers leaked in.

> [!PAPER] · Table 10
> [![Table 10 of the paper: OpenMathReasoning 281,743 problems, DeepMath-103K 103,021, NuminaMath-1.5 129,520, PolyMath 11,083, Omni-MATH 4,289, FineProofs-SFT 4,275, AIME 1983 to 2024 933, Putnam-AXIOM 492, total 535,356, with 22 percent proof problems](/img/papers/meta-harness/p4-table10.png)](/img/papers/meta-harness/p4-table10.png)
>
> **What it says:** "Datasets in the math retrieval corpus (535K problems total)". The table gives the problem count, the median solution length and the share of proof problems for each of the eight sources.
>
> **What it means:**
>
> Two sources make up most of the corpus: OpenMathReasoning (281,743 problems) and NuminaMath-1.5 (129,520). The small sources are the most like the test: 933 old AIME problems and 492 Putnam problems.
>
> The "Proof" column matters. A proof problem asks "show that ...", and its solution is an argument, not a number. In total 22% of the corpus are proof problems. Long solutions are cut at 5,000 characters, so a prompt with three examples stays a manageable size.

> [!PAPER] · Appendix C.2
> [![Appendix C.2 of the paper, with exact prefix matching followed by fuzzy Jaccard similarity threshold 0.8, truncated again to 3,000 characters, and difficulty greater than 6 highlighted](/img/papers/meta-harness/p4-c2.png)](/img/papers/meta-harness/p4-c2.png)
>
> **What it says:** the corpus was decontaminated "using exact prefix matching followed by fuzzy Jaccard similarity (threshold 0.8)". At run time the harness keeps solutions shorter than 4,000 characters, and inserts each one "truncated again to 3,000 characters". The geometry route has its own index of NuminaMath problems "with difficulty greater than 6".
>
> **What it means:**
>
> Two tests decide if a corpus problem is too close to a test problem. The first is exact: do the two texts start with the same characters? The second is fuzzy: do the two problems use almost the same set of words? The fuzzy test is the **Jaccard similarity** of the two word sets, and 0.8 is the cut-off.
>
> The last sentence is a detail that comes back later. The harness keeps a second, small index with only hard geometry problems. One route of the discovered harness always takes one example from it.

> [!DEFINITION] Jaccard similarity
> For two sets $$A$$ and $$B$$, the size of what they share divided by the size of everything in either: $$J(A, B) = |A \cap B| \, / \, |A \cup B|$$. Two identical sets give 1. Two sets with nothing in common give 0.

Here is the Jaccard test on a tiny pair. "find x if 2x = 10" has the word set {find, x, if, 2x, 10}. "find x if 2x = 12" has {find, x, if, 2x, 12}. They share 4 words, and together they have 6, so $$J = 4/6 = 0.67$$. That is below 0.8, so the filter keeps the second problem. A short problem with one new number can pass the filter, which is why the authors also read the top search results by hand.

### Decontamination in code

We cannot download the paper's corpus and test set at laptop scale, so we use a smaller public set with the same shape. **MATH** has 12,500 competition problems with full solutions. Each has a subject label (Algebra, Geometry, Number Theory and four more) and a difficulty level from 1 to 5. Its 7,500 training problems are our corpus, and its 5,000 test problems are our queries.

Our filter does both of the paper's tests. Exact prefix: the first 60 characters, in lower case. Fuzzy: the Jaccard similarity of the word sets, with the same 0.8 cut-off.

```python
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
```

The last line uses a fact about sets: the size of the union is $$|A| + |B| - |A \cap B|$$. So once we count the shared words, we get the Jaccard similarity of every corpus problem without building a single union.

```text
== 4. Decontamination (Appendix C.2): prefix match or word Jaccard >= 0.8
491 of 5000 test problems match a corpus problem (49 s)
  e.g. If $2^8=4^x$, what is the value of $x$?
  e.g. What is the number of units in the distance from the origin to the point $(-8,6)$ in a coo
```

This surprised us. Almost one test problem in ten has a near copy in the training half of MATH. Without the filter, a retriever would fetch the answer for those 491 problems, and any score would look better than it is. Every experiment below uses only the 4,509 clean test problems.

## The search and the test {§4.2}

> [!PAPER] · §4.2
> [![The paragraph on the search, with 40 iterations over a 250-problem search set, 109 candidate retrieval harnesses, select a single harness based on search-set performance using GPT-OSS-20B, 200 previously unseen IMO-level problems, four models not seen during search, and report accuracy averaged over three samples per problem highlighted](/img/papers/meta-harness/p4-search.png)](/img/papers/meta-harness/p4-search.png)
>
> **What it says:** Meta-Harness ran "40 iterations over a 250-problem search set" and produced "109 candidate retrieval harnesses". The authors "select a single harness based on search-set performance using GPT-OSS-20B". They test it on "200 previously unseen IMO-level problems" and on "four models not seen during search", and "report accuracy averaged over three samples per problem".
>
> **What it means:**
>
> Read this as an exam with strict rules. During the search, the proposer (the coding agent that writes new harnesses) sees only the 250 search problems. One model, GPT-OSS-20B, solves them with each candidate harness. The best candidate on those 250 problems is the one harness that goes forward.
>
> Then the authors change two things at once. The test problems are new and much harder: IMO level, the International Mathematical Olympiad. And four of the five test models were never used during the search. If the harness still helps, it helps because its policy is good, not because it was tuned to one model.
>
> The last phrase is about noise. A model that samples its answer can solve a problem on one try and fail on the next. So each problem gets three tries, and the score is the average.

> [!DEFINITION] Search set and test set
> The **search set** is the data the search loop may score candidates on. The **test set** is kept apart and used once, at the end. A harness that wins on the search set and also wins on the test set has learned something that carries over.

> [!DEFINITION] Pass@1
> The share of problems a model solves with one answer per problem. "pass@1 averaged over three samples" means: draw three answers per problem, find the share correct for each draw, and average the three shares.

> [!PAPER] · Table 11
> [![Table 11 of the paper: IMO-AnswerBench 100 problems, IMO-ProofBench 60, ArXivMath December 2025 17, ArXivMath January 2026 23, total 200](/img/papers/meta-harness/p4-table11.png)](/img/papers/meta-harness/p4-table11.png)
>
> **What it says:** the "200-problem IMO-level evaluation set" has 100 IMO-AnswerBench problems, 60 IMO-ProofBench problems, and 17 plus 23 ArXivMath problems from December 2025 and January 2026.
>
> **What it means:**
>
> The mix is deliberate. IMO-AnswerBench problems have a short final answer. IMO-ProofBench problems need a proof. The ArXivMath problems are dated December 2025 and January 2026, a few months before the paper. They are so new that they are unlikely to be in any corpus, so they test the harness on unseen material.

## BM25: the search engine under every route

Every harness in this section, the discovered one too, stands on one classic search method: **BM25**. It dates from the 1990s, and it is still the first thing to try when you search text by its words. It is worth understanding from the inside.

> [!DEFINITION] Lexical retrieval and BM25
> **Lexical retrieval** scores a document by the words it shares with the query. It does not know that "prime" and "divisible" are related; it only sees matching words. **BM25** ("Best Match 25") is the standard lexical scoring formula.

> [!DEFINITION] Token
> One unit of text that the search engine compares: a word, a number or a LaTeX command. Splitting text into tokens is called tokenizing.

BM25 follows three common-sense rules. A rare word is better evidence than a common one: "remainder" tells you more than "the". A word that occurs more often in a document is better evidence, but the tenth time adds much less than the first. And a hit in a short document counts more than a hit in a long one, because a long document has many words by chance.

$$
\text{score}(D, Q) = \sum_{q \in Q} \text{idf}(q) \cdot \frac{f(q, D)\,(k_1 + 1)}{f(q, D) + k_1 \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
\qquad
\text{idf}(q) = \ln \frac{N - n(q) + 0.5}{n(q) + 0.5}
$$

Read it from left to right:

- $$D$$ is one document and $$Q$$ is the query, a list of tokens. The score is a sum with one term for each query token $$q$$.
- $$\text{idf}(q)$$ is the **inverse document frequency**: how rare the token is. $$N$$ is the number of documents, and $$n(q)$$ is the number that contain $$q$$. A token in few documents gets a large idf.
- $$f(q, D)$$ is how many times $$q$$ occurs in $$D$$, the **term frequency**.
- $$|D|$$ is the length of $$D$$ in tokens, and $$\text{avgdl}$$ is the average length over the corpus.
- $$k_1$$ controls how fast extra hits stop helping. $$b$$ controls how much length matters. The usual values are $$k_1 = 1.5$$ and $$b = 0.75$$.

When a token is in more than half the documents, the idf formula gives a negative number. The common library version then puts a small positive floor in its place: a quarter of the average idf. Our code does the same, so the two agree.

### A worked example by hand

Take five tiny documents and the query "prime remainder":

1. "find the remainder when 100 is divided by 7" (9 tokens)
2. "the prime 7 and the prime 11 and the prime 13" (11 tokens)
3. "the radius of the circle is 6" (7 tokens)
4. "is 91 a prime" (4 tokens)
5. "the area of the square is 36" (7 tokens)

The average length is $$38 / 5 = 7.6$$. "prime" is in 2 of the 5 documents, so $$\text{idf} = \ln(3.5 / 2.5) = 0.3365$$. "remainder" is in only 1, so $$\text{idf} = \ln(4.5 / 1.5) = 1.0986$$. The rare word is worth three times as much.

Document 1 has "remainder" once and 9 tokens. The length part is $$1.5 \times (0.25 + 0.75 \times 9 / 7.6) = 1.7072$$. The term part is $$2.5 / (1 + 1.7072) = 0.9235$$, and the score is $$1.0986 \times 0.9235 = 1.0145$$.

Document 2 says "prime" three times, in 11 tokens. Its term part is $$7.5 / (3 + 2.0033) = 1.4990$$. Document 4 says "prime" once, but it is very short, so its term part is already 1.2709. Three hits beat one hit by only 18%. The score of document 2 is $$0.3365 \times 1.4990 = 0.5044$$, and document 4 gets 0.4276. Document 1 wins: one rare word beats three common ones.

### Method 1: BM25 from scratch

The code builds an **inverted index**: for each token, the list of documents that contain it and how often. A query then only touches the documents that share a token with it, which is how real search engines stay fast.

```python
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
```

### Method 2: the rank_bm25 library, and the check

`rank_bm25` is a small, widely used Python package with the same formula in its class `BM25Okapi`. We give both methods the same five documents:

```python
    from rank_bm25 import BM25Okapi
    lib = BM25Okapi(toks)
    s2 = lib.get_scores(['prime', 'remainder'])
```

```text
== 1. BM25 by hand: five tiny documents, query "prime remainder"
N = 5, lengths = [9, 11, 7, 4, 7], avgdl = 7.600
idf(prime) = 0.3365   (in 2 of 5 docs)
idf(remainder) = 1.0986   (in 1 of 5 docs)
idf(the) = 0.1987   (in 4 of 5 docs)
doc 2: tf(prime) = 3, length 11, k1 * (1 - b + b * dl/avgdl) = 2.0033, tf part = 1.4990
doc 4: tf(prime) = 1, length 4, k1 * (1 - b + b * dl/avgdl) = 0.9671, tf part = 1.2709
tf part, dl = 0.5 x avgdl, tf = 1..10: [1.29, 1.7, 1.9, 2.03, 2.11, 2.16, 2.2, 2.24, 2.26, 2.29]
tf part, dl = 1.0 x avgdl, tf = 1..10: [1.0, 1.43, 1.67, 1.82, 1.92, 2.0, 2.06, 2.11, 2.14, 2.17]
tf part, dl = 2.0 x avgdl, tf = 1..10: [0.69, 1.08, 1.33, 1.51, 1.64, 1.74, 1.82, 1.88, 1.94, 1.98]
scores: [1.0145, 0.5044, 0.0, 0.4276, 0.0]
rank_bm25: [1.0145, 0.5044, 0.0, 0.4276, 0.0] max diff 1.1e-16
```

The scores are the ones we found by hand, and the library agrees to $$10^{-16}$$. "the" is in 4 of 5 documents, so its raw idf is negative and it gets the floor, 0.1987. The three "tf part" lines are the saturation rule in numbers; the chart shows them.

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="The term-frequency part of BM25 with k1 = 1.5 and b = 0.75, for 1 to 10 hits of one word. It grows fast at first and then flattens toward 2.5. A short document gets more credit per hit than a long one."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="500" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="206.0" x2="500" y2="206.0"/><text class="t-tick" x="56.0" y="210.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="162.0" x2="500" y2="162.0"/><text class="t-tick" x="56.0" y="166.0" text-anchor="end">1.00</text><line class="grid" x1="64" y1="118.0" x2="500" y2="118.0"/><text class="t-tick" x="56.0" y="122.0" text-anchor="end">1.50</text><line class="grid" x1="64" y1="74.0" x2="500" y2="74.0"/><text class="t-tick" x="56.0" y="78.0" text-anchor="end">2.00</text><line class="grid" x1="64" y1="30.0" x2="500" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">2.50</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1</text><text class="t-tick" x="112.4" y="270.0" text-anchor="middle">2</text><text class="t-tick" x="160.9" y="270.0" text-anchor="middle">3</text><text class="t-tick" x="209.3" y="270.0" text-anchor="middle">4</text><text class="t-tick" x="257.8" y="270.0" text-anchor="middle">5</text><text class="t-tick" x="306.2" y="270.0" text-anchor="middle">6</text><text class="t-tick" x="354.7" y="270.0" text-anchor="middle">7</text><text class="t-tick" x="403.1" y="270.0" text-anchor="middle">8</text><text class="t-tick" x="451.6" y="270.0" text-anchor="middle">9</text><text class="t-tick" x="500.0" y="270.0" text-anchor="middle">10</text><line class="axis" x1="64" y1="250" x2="500" y2="250"/><polyline class="l1" points="64.0,136.5 112.4,100.2 160.9,82.4 209.3,71.8 257.8,64.7 306.2,59.7 354.7,56.0 403.1,53.1 451.6,50.8 500.0,48.9"/><g class="mark"><title>short doc (half the average length), 1: 1.29</title><circle class="s1 ring" cx="64.0" cy="136.5" r="5"/></g><g class="mark"><title>short doc (half the average length), 2: 1.70</title><circle class="s1 ring" cx="112.4" cy="100.2" r="5"/></g><g class="mark"><title>short doc (half the average length), 3: 1.90</title><circle class="s1 ring" cx="160.9" cy="82.4" r="5"/></g><g class="mark"><title>short doc (half the average length), 4: 2.03</title><circle class="s1 ring" cx="209.3" cy="71.8" r="5"/></g><g class="mark"><title>short doc (half the average length), 5: 2.11</title><circle class="s1 ring" cx="257.8" cy="64.7" r="5"/></g><g class="mark"><title>short doc (half the average length), 6: 2.16</title><circle class="s1 ring" cx="306.2" cy="59.7" r="5"/></g><g class="mark"><title>short doc (half the average length), 7: 2.20</title><circle class="s1 ring" cx="354.7" cy="56.0" r="5"/></g><g class="mark"><title>short doc (half the average length), 8: 2.24</title><circle class="s1 ring" cx="403.1" cy="53.1" r="5"/></g><g class="mark"><title>short doc (half the average length), 9: 2.26</title><circle class="s1 ring" cx="451.6" cy="50.8" r="5"/></g><g class="mark"><title>short doc (half the average length), 10: 2.29</title><circle class="s1 ring" cx="500.0" cy="48.9" r="5"/></g><polyline class="l2" points="64.0,162.0 112.4,124.3 160.9,103.3 209.3,90.0 257.8,80.8 306.2,74.0 354.7,68.8 403.1,64.7 451.6,61.4 500.0,58.7"/><g class="mark"><title>average-length doc, 1: 1.00</title><circle class="s2 ring" cx="64.0" cy="162.0" r="5"/></g><g class="mark"><title>average-length doc, 2: 1.43</title><circle class="s2 ring" cx="112.4" cy="124.3" r="5"/></g><g class="mark"><title>average-length doc, 3: 1.67</title><circle class="s2 ring" cx="160.9" cy="103.3" r="5"/></g><g class="mark"><title>average-length doc, 4: 1.82</title><circle class="s2 ring" cx="209.3" cy="90.0" r="5"/></g><g class="mark"><title>average-length doc, 5: 1.92</title><circle class="s2 ring" cx="257.8" cy="80.8" r="5"/></g><g class="mark"><title>average-length doc, 6: 2.00</title><circle class="s2 ring" cx="306.2" cy="74.0" r="5"/></g><g class="mark"><title>average-length doc, 7: 2.06</title><circle class="s2 ring" cx="354.7" cy="68.8" r="5"/></g><g class="mark"><title>average-length doc, 8: 2.11</title><circle class="s2 ring" cx="403.1" cy="64.7" r="5"/></g><g class="mark"><title>average-length doc, 9: 2.14</title><circle class="s2 ring" cx="451.6" cy="61.4" r="5"/></g><g class="mark"><title>average-length doc, 10: 2.17</title><circle class="s2 ring" cx="500.0" cy="58.7" r="5"/></g><polyline class="l3" points="64.0,189.3 112.4,154.9 160.9,132.7 209.3,117.2 257.8,105.7 306.2,97.0 354.7,90.0 403.1,84.4 451.6,79.7 500.0,75.7"/><g class="mark"><title>long doc (twice the average), 1: 0.69</title><circle class="s3 ring" cx="64.0" cy="189.3" r="5"/></g><g class="mark"><title>long doc (twice the average), 2: 1.08</title><circle class="s3 ring" cx="112.4" cy="154.9" r="5"/></g><g class="mark"><title>long doc (twice the average), 3: 1.33</title><circle class="s3 ring" cx="160.9" cy="132.7" r="5"/></g><g class="mark"><title>long doc (twice the average), 4: 1.51</title><circle class="s3 ring" cx="209.3" cy="117.2" r="5"/></g><g class="mark"><title>long doc (twice the average), 5: 1.64</title><circle class="s3 ring" cx="257.8" cy="105.7" r="5"/></g><g class="mark"><title>long doc (twice the average), 6: 1.74</title><circle class="s3 ring" cx="306.2" cy="97.0" r="5"/></g><g class="mark"><title>long doc (twice the average), 7: 1.82</title><circle class="s3 ring" cx="354.7" cy="90.0" r="5"/></g><g class="mark"><title>long doc (twice the average), 8: 1.88</title><circle class="s3 ring" cx="403.1" cy="84.4" r="5"/></g><g class="mark"><title>long doc (twice the average), 9: 1.94</title><circle class="s3 ring" cx="451.6" cy="79.7" r="5"/></g><g class="mark"><title>long doc (twice the average), 10: 1.98</title><circle class="s3 ring" cx="500.0" cy="75.7" r="5"/></g><line class="grid" x1="506.0" y1="48.9" x2="516.0" y2="48.9"/><rect class="s1" x="520.0" y="43.9" width="10" height="10" rx="2"/><text class="t-note" x="536.0" y="52.9" text-anchor="start">short doc (half the average length): 2.29</text><line class="grid" x1="506.0" y1="58.7" x2="516.0" y2="65.9"/><rect class="s2" x="520.0" y="60.9" width="10" height="10" rx="2"/><text class="t-note" x="536.0" y="69.9" text-anchor="start">average-length doc: 2.17</text><line class="grid" x1="506.0" y1="75.7" x2="516.0" y2="82.9"/><rect class="s3" x="520.0" y="77.9" width="10" height="10" rx="2"/><text class="t-note" x="536.0" y="86.9" text-anchor="start">long doc (twice the average): 1.98</text><text class="t-tick" x="282.0" y="290.0" text-anchor="middle">times the word occurs in the document (tf)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">tf part of the score</text></svg><figcaption>The term-frequency part of BM25 for 1 to 10 hits of one word, for a short, an average and a long document. The first hit adds the most. After about five hits the curve is almost flat, and it never goes above 2.5. At equal hits, the short document always scores higher.</figcaption></figure>

On the full MATH corpus, the two methods also agree:

```text
== 3. The corpus: MATH train = corpus, MATH test = queries
corpus 7500 problems, test 5000 problems (23.8 s to load)
test subjects: {'Algebra': 1187, 'Counting & Probability': 474, 'Geometry': 479, 'Intermediate Algebra': 903, 'Number Theory': 540, 'Prealgebra': 871, 'Precalculus': 546}
Method 1 (ours): indexed in 1.2 s, 7501 distinct tokens, avgdl 42.2
Method 2 (rank_bm25): indexed in 0.8 s
20 queries: largest score difference 3.6e-12; same top-5 list for 20 of 20
```

### Tokens for math

The paper says its BM25 index "uses a math-aware tokenizer that preserves LaTeX tokens (e.g., \frac, ^{2}) as atomic units". The reason is easy to see in code. A naive tokenizer keeps only letters and digits. The LaTeX command `\frac` becomes the word "frac", and the square in `x^{2}` becomes a lone "2" that looks like any other 2.

```python
MATH_TOKEN = re.compile(r'\\[A-Za-z]+'          # a LaTeX command: \frac, \sqrt, \angle
                        r'|[\^_]\{[^{}]*\}'      # a superscript or subscript group: ^{2}, _{10}
                        r'|[\^_]\w'              # a short one: ^2, _7
                        r'|\d+'                  # a number
                        r'|[A-Za-z]+')           # a word


def tokenize_math(text):
    """Math-aware tokens: every LaTeX command and every ^{...} or _{...} group stays one token."""
    return [t if t.startswith(('\\', '^', '_')) else t.lower() for t in MATH_TOKEN.findall(text)]
```

```text
== 2. Tokenizers on one problem
text: Compute $\frac{1}{2}+x^{2}$ where $x^2 = 9$ and $\angle ABC = 90^\circ$.
naive: ['compute', 'frac', '1', '2', 'x', '2', 'where', 'x', '2', '9', 'and', 'angle', 'abc', '90', 'circ']
math : ['compute', '\\frac', '1', '2', 'x', '^{2}', 'where', 'x', '^2', '9', 'and', '\\angle', 'abc', '90', '\\circ']
```

With the math tokens, "squared" is its own token, and `\angle` is not confused with the English word "angle". On MATH the gain is small, as we show below. On a corpus of half a million problems, written in many LaTeX styles, the authors' search kept it.

## The discovered harness: a router and four policies {Appendix B.2}

> [!PAPER] · Appendix B.2
> [![The first lines of Appendix B.2, with compact four-route BM25 program and selected by the outer loop across 40 iterations highlighted](/img/papers/meta-harness/p4-b2-intro.png)](/img/papers/meta-harness/p4-b2-intro.png)
>
> **What it says:** the final harness is "a compact four-route BM25 program". The routing rules, the reranking terms, the deduplication thresholds and the number of examples per route "were selected by the outer loop across 40 iterations of evolution".
>
> **What it means:**
>
> The authors want to be clear about who designed what. No person wrote these rules and then called them a discovery. The structure came out of the search loop, and the appendix only describes the program that won.

> [!PAPER] · Appendix B.2
> [![The Overview paragraph of Appendix B.2, with exactly one of four routes, lightweight lexical predicates, math-aware tokenizer and merge of two successful search lineages highlighted](/img/papers/meta-harness/p4-b2-overview.png)](/img/papers/meta-harness/p4-b2-overview.png)
>
> **What it says:** the harness sends each problem to "exactly one of four routes": combinatorics, geometry, number theory, or a default route for algebra and the rest. The gates are "lightweight lexical predicates": keyword sets and a few regex features for geometry notation. All routes use BM25. The final harness is a "merge of two successful search lineages": one gave a better geometry route, the other a better combinatorics route.
>
> **What it means:**
>
> Think of the front desk of a hospital. The nurse reads a few words on your form and sends you to one department. Each department then works in its own way. Here, the "form" is the problem text, and the nurse is a few lines of code that look for words such as "probability" or "remainder".
>
> The last sentence is the most interesting one. In the search, two families of harnesses grew apart. One got good at geometry, the other at combinatorics. The proposer read both, saw which route each did best, and wrote one program that takes the best route from each. A person does this when they merge two branches of code. An optimizer that sees only scores cannot do it, because it does not know which part of a program earned the score.

> [!DEFINITION] Router and predicate
> A **router** is code that sends each input to one of several paths. A **predicate** is a test that answers yes or no, such as "does the problem contain the word *probability*?". A **lexical** predicate looks only at the words.

> [!DEFINITION] Regex
> A regular expression: a short pattern that matches text. For example, `\b[A-Z]{3}\b` matches three capital letters on their own, such as the ABC in "triangle ABC".

> [!PAPER] · Appendix B.2
> [![The four route bullets of Appendix B.2, with fetch 20 BM25 candidates deduplicate to 8 rerank by lexical score and difficulty, return 1 hard NuminaMath reference together with 2 raw BM25 neighbors, small bonus for solutions that state a technique early, and adaptive number of examples highlighted](/img/papers/meta-harness/p4-b2-routes.png)](/img/papers/meta-harness/p4-b2-routes.png)
>
> **What it says:** Combinatorics: "fetch 20 BM25 candidates, deduplicate to 8, rerank by lexical score and difficulty", then keep 3. Geometry: "return 1 hard NuminaMath reference together with 2 raw BM25 neighbors". Number theory: fetch 12 and rerank with "a small bonus for solutions that state a technique early". Default: fetch 10, rerank, and pick an "adaptive number of examples".
>
> **What it means:**
>
> Each route encodes a guess about its subject, and each guess makes sense once you see it. Counting problems are often near copies of each other ("how many ways to seat 5 people" and "6 people"), so the combinatorics route removes repeats and prefers harder problems. That gives the model three different ideas, not one idea three times.
>
> Geometry problems look alike on the surface (a triangle, a circle) but need different ideas. Here the search kept the raw BM25 order, because it "consistently prefers raw structural matches". It also adds one hard reference problem, as a model of a careful solution.
>
> Number theory rewards a solution that names its method early, such as "work modulo 9". A model can copy a method only when the example states it. The default route decides how many examples to show. It shows one if the top match stands far above the rest, and more if several are about equally good.

> [!DEFINITION] Rerank
> Retrieve a list of candidates with a fast score, then sort that short list again with a better score. Here the new score mixes the BM25 score with the difficulty of the problem, and sometimes with a bonus for a named technique.

> [!PAPER] · Figure 8
> [![Figure 8 of the paper: a Query box goes to a Lexical router with keyword and regex cues. Four arrows go to Combinatorics (BM25 at 20, dedup to 8, rerank, keep 3), Geometry (1 fixed ref plus 2 BM25, no rerank), Number theory (BM25 at 12, rerank, keep 3) and Algebra or Other (BM25 at 10, rerank, adaptive K). All four go to Build final prompt](/img/papers/meta-harness/p4-figure8.png)](/img/papers/meta-harness/p4-figure8.png)
>
> **What it says:** "A lexical router assigns each query to one of four subject-specific retrieval policies. The selected policy retrieves examples, which are inserted into the final prompt."
>
> **What it means:**
>
> Read the figure from the top. One query comes in. The router picks exactly one of the four boxes, so the other three do nothing for this query. The chosen box runs its own recipe, and its examples go into the prompt.
>
> "BM25@20" means "the top 20 BM25 results". "Adaptive K" means the number of examples is decided per query. Note what is absent: there is no second model, no learned embedding, and no voting between routes. The whole discovered harness is ordinary code over one BM25 index.

### Our router, from scratch

The paper does not print its keyword lists, so we wrote our own: about 20 keywords per route and six regex cues for geometry notation. The full lists are in `mh_part4.py`. The router counts the cues of each route, and the route with the most cues wins.

```python
GEOMETRY_REGEX = [r'\\angle', r'\\triangle', r'\\overline\{[A-Z]{2}\}', r'\\pi r', r'\b[A-Z]{3}\b',
                  r'\b[A-Z]{2}\s*=\s*\d']        # notation such as triangle ABC, AB = 5


def route(problem):
    """Count the cues of each route; the route with the most cues wins. No cue at all: the default route."""
    text = problem.lower()
    hits = {r: sum(text.count(k) for k in ks) for r, ks in KEYWORDS.items()}
    hits['geometry'] += sum(len(re.findall(p, problem)) for p in GEOMETRY_REGEX)
    best = max(hits, key=hits.get)
    return best if hits[best] > 0 else 'default'
```

MATH gives us something the paper's test set does not: a subject label on every problem. So we can grade the router. Counting & Probability should go to combinatorics, Geometry to geometry, Number Theory to number theory, and the other four subjects to default.

```text
== 5. The lexical router, scored against the dataset's subject labels
train: 7500 problems, router accuracy 76.4%
test: 4509 problems, router accuracy 77.6%
    true route | combinat geometry number_t  default | correct
 combinatorics |      257       45       43       80 |  60.5%
      geometry |        6      369        5       63 |  83.3%
 number_theory |       12       17      415       70 |  80.7%
       default |       78      385      206     2458 |  78.6%
  route('How many ways can 4 books be arranged on a shelf?') = combinatorics
  route('What is the remainder when $2^{100}$ is divided by 7?') = number_theory
  route('In triangle $ABC$, $AB = 5$ and $\\angle C = 90^\\circ$. Find $BC$.') = geometry
  route('Solve for $x$: $3x + 7 = 22$.') = default
```

<figure class="fig"><svg viewBox="0 0 460 440" role="img" aria-label="Confusion table of our lexical router on 4,509 MATH test problems. Each row is a true route and adds up to 1. The diagonal holds 0.61, 0.83, 0.81 and 0.79."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="132.0" y="89.0" text-anchor="end">combinatorics</text><g class="mark"><title>combinatorics → combinatorics: 0.60</title><rect class="cell" x="141.0" y="51.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.605"/></g><text class="t-cell on" x="175.0" y="89.0" text-anchor="middle">0.60</text><g class="mark"><title>combinatorics → geometry: 0.11</title><rect class="cell" x="211.0" y="51.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.106"/></g><text class="t-cell" x="245.0" y="89.0" text-anchor="middle">0.11</text><g class="mark"><title>combinatorics → number theory: 0.10</title><rect class="cell" x="281.0" y="51.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.101"/></g><text class="t-cell" x="315.0" y="89.0" text-anchor="middle">0.10</text><g class="mark"><title>combinatorics → default: 0.19</title><rect class="cell" x="351.0" y="51.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.188"/></g><text class="t-cell" x="385.0" y="89.0" text-anchor="middle">0.19</text><text class="t-tick" x="132.0" y="159.0" text-anchor="end">geometry</text><g class="mark"><title>geometry → combinatorics: 0.01</title><rect class="cell" x="141.0" y="121.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-cell" x="175.0" y="159.0" text-anchor="middle">0.01</text><g class="mark"><title>geometry → geometry: 0.83</title><rect class="cell" x="211.0" y="121.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.833"/></g><text class="t-cell on" x="245.0" y="159.0" text-anchor="middle">0.83</text><g class="mark"><title>geometry → number theory: 0.01</title><rect class="cell" x="281.0" y="121.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-cell" x="315.0" y="159.0" text-anchor="middle">0.01</text><g class="mark"><title>geometry → default: 0.14</title><rect class="cell" x="351.0" y="121.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.142"/></g><text class="t-cell" x="385.0" y="159.0" text-anchor="middle">0.14</text><text class="t-tick" x="132.0" y="229.0" text-anchor="end">number theory</text><g class="mark"><title>number theory → combinatorics: 0.02</title><rect class="cell" x="141.0" y="191.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-cell" x="175.0" y="229.0" text-anchor="middle">0.02</text><g class="mark"><title>number theory → geometry: 0.03</title><rect class="cell" x="211.0" y="191.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-cell" x="245.0" y="229.0" text-anchor="middle">0.03</text><g class="mark"><title>number theory → number theory: 0.81</title><rect class="cell" x="281.0" y="191.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.807"/></g><text class="t-cell on" x="315.0" y="229.0" text-anchor="middle">0.81</text><g class="mark"><title>number theory → default: 0.14</title><rect class="cell" x="351.0" y="191.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.136"/></g><text class="t-cell" x="385.0" y="229.0" text-anchor="middle">0.14</text><text class="t-tick" x="132.0" y="299.0" text-anchor="end">default</text><g class="mark"><title>default → combinatorics: 0.02</title><rect class="cell" x="141.0" y="261.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-cell" x="175.0" y="299.0" text-anchor="middle">0.02</text><g class="mark"><title>default → geometry: 0.12</title><rect class="cell" x="211.0" y="261.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.123"/></g><text class="t-cell" x="245.0" y="299.0" text-anchor="middle">0.12</text><g class="mark"><title>default → number theory: 0.07</title><rect class="cell" x="281.0" y="261.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.066"/></g><text class="t-cell" x="315.0" y="299.0" text-anchor="middle">0.07</text><g class="mark"><title>default → default: 0.79</title><rect class="cell" x="351.0" y="261.0" width="68.0" height="68.0" rx="2" style="fill-opacity:0.786"/></g><text class="t-cell on" x="385.0" y="299.0" text-anchor="middle">0.79</text><text class="t-tick" x="175.0" y="340.0" text-anchor="end" transform="rotate(-50 175.0 340.0)">combinatorics</text><text class="t-tick" x="245.0" y="340.0" text-anchor="end" transform="rotate(-50 245.0 340.0)">geometry</text><text class="t-tick" x="315.0" y="340.0" text-anchor="end" transform="rotate(-50 315.0 340.0)">number theory</text><text class="t-tick" x="385.0" y="340.0" text-anchor="end" transform="rotate(-50 385.0 340.0)">default</text><text class="t-muted" x="280.0" y="30.0" text-anchor="middle">route chosen by the router</text><text class="t-muted" x="132.0" y="40.0" text-anchor="end">true route</text></svg><figcaption>The confusion table of our router on 4,509 clean MATH test problems. Each row is the true route and adds up to 1; each column is the route the router chose. The diagonal is the share it got right. The largest error is in the bottom row: 385 algebra, precalculus and other problems went to geometry.</figcaption></figure>

A keyword router gets about three problems in four right, and it is wrong in ways you can read. Many precalculus problems talk about angles and circles, so they go to geometry. A counting problem that never says "ways" or "probability" falls to default. A wrong route is not a disaster here, because every route still runs BM25 on the whole corpus. It changes how many examples come back and how they are sorted, not where they come from.

### The four policies in code

The four recipes of Figure 8 fit in one function. The paper does not give its weights or thresholds, so ours are our own choice. The rerank adds 0.15 for difficulty and 0.1 for a named technique. MATH's level (1 to 5) stands in for difficulty, and our hard geometry index holds the 421 level-5 geometry problems.

```python
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
```

Here it is on one test problem of each route:

```text
== 6. The routed harness on one test problem per route
hard geometry index: 421 level-5 geometry problems
[combinatorics] query: 'In how many ways can we choose 3 distinct letters of the alphabet, without regar'
    -> Prealgebra, level 5: 'Acme Corporation has released an alphabet soup in which each'
    -> Prealgebra, level 5: 'In how many ways can a president and a 2-person committee be'
    -> Counting & Probability, level 5: 'I have a bag with $6$ marbles numbered from $1$ to $6.$ Math'
[default] query: 'Evaluate $\\log_{5^2}5^4$.'
    -> Algebra, level 3: 'Evaluate $\\log_{3}{81}-\\log_{3}{\\frac{1}{9}}$.'
    -> Algebra, level 3: 'Evaluate $\\log_432$.'
    -> Algebra, level 2: 'Evaluate $\\log_5625$.'
[number_theory] query: 'What is the last nonzero digit to the right of the decimal point in the decimal '
    -> Number Theory, level 3: 'What is the last digit of the decimal expansion of $\\frac{1}'
    -> Number Theory, level 2: 'What is the first nonzero digit to the right of the decimal '
    -> Prealgebra, level 3: 'What is the 100th digit to the right of the decimal point in'
[geometry] query: 'In the diagram, what is the perimeter of the sector of the circle with radius 12'
    -> Geometry, level 5: 'The base of a triangular piece of paper $ABC$ is $12\\text{ c'
    -> Geometry, level 4: 'In the diagram, $AOB$ is a sector of a circle with $\\angle A'
    -> Geometry, level 2: 'Two sectors of a circle of radius $12$ are placed side by si'
```

The examples are close in method. The logarithm question gets three logarithm questions. The decimal-digit question gets two problems about digits of a decimal expansion. The combinatorics route lifted two level-5 problems to the top, because of the difficulty bonus. Both are counting problems, even though MATH files them under Prealgebra.

One more test: how often do the retrieved examples come from the query's own subject? We ask this for 400 clean test problems.

```text
== 7. Do the retrieved examples share the query's subject? (400 test problems)
 bm25 top-3 (naive tokens): 62.0% of retrieved examples share the subject
  bm25 top-3 (math tokens): 62.6% of retrieved examples share the subject
            routed harness: 62.8% of retrieved examples share the subject
routed harness: 2.83 examples per problem on average (76 s for 400 queries x 3 methods)
```

The three numbers are almost the same. That is an honest result, and it teaches something. Plain BM25 already finds problems of the same subject most of the time, because the words give the subject away. The router does not change *where* the examples come from. It changes *which* of the near neighbours are shown, how hard they are, and how many. Whether that helps can only be measured by a model that reads them, as the paper does next.

[![Terminal output of mh_part4.py with every retrieval result on this page](/img/papers/meta-harness/part4-run.png)](/img/papers/meta-harness/part4-run.png)

## The results {§4.2, Table 6}

> [!PAPER] · Table 6
> [![Table 6 of the paper: pass at 1 of five models, GPT-5.4 nano, GPT-5.4 mini, Gemini 3.1 Flash-Lite, Gemini 3 Flash and GPT-OSS-20B, for no retriever, dense retrieval with k equal 1 and 5, random few-shot, BM25 retrieval and Meta-Harness. The averages are 34.1, 34.4, 38.1, 32.2, 37.5 and 38.8](/img/papers/meta-harness/p4-table6.png)](/img/papers/meta-harness/p4-table6.png)
>
> **What it says:** on 200 IMO-level problems, with "pass@1 averaged over three samples per problem", the discovered harness improves every one of the five held-out models, "with a 4.7-point average gain over no retriever".
>
> **What it means:**
>
> Read the table one row at a time. Without any retrieval, the five models average 34.1%. Random few-shot (three random solved problems in the prompt) makes things worse: 32.2%. So examples alone do not help; the wrong examples cost the model attention and push it off course.
>
> Dense retrieval with one example barely moves the average (34.4%), and with five it reaches 38.1%. But look at the red numbers: dense retrieval hurts GPT-5.4 mini and GPT-OSS-20B. BM25 reaches 37.5%. The discovered harness reaches 38.8%, and it has no red number in its row. Every one of the five models gets better with it.
>
> The gains are modest. On the weakest model, GPT-5.4 nano, it adds 8.7 points. On GPT-5.4 mini it adds only 1.6. A gain of 4.7 points on average is real, but it does not turn a weak model into an olympiad medalist.

> [!DEFINITION] Dense retrieval and embedding
> An **embedding** is a list of numbers that a neural network computes for a text, so that texts with similar meaning get similar lists. **Dense retrieval** compares those lists, not the words. It can match "prime" with "divisible", but it needs a second model; here, text-embedding-3-small.

<figure class="fig"><svg viewBox="0 0 680 240" role="img" aria-label="Bar chart of the average column of Table 6: no retriever 34.1, dense k=1 34.4, dense k=5 38.1, random few-shot 32.2, BM25 37.5, Meta-Harness 38.8 percent."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="30.0" text-anchor="start">Table 6: average pass@1 over five held-out models</text><text class="t-tick" x="162.0" y="59.0" text-anchor="end">No retriever</text><g class="mark"><title>No retriever: 34.1%</title><rect class="s1" x="170.0" y="43.0" width="318.3" height="22.0" rx="3"/></g><text class="t-val" x="494.3" y="59.0" text-anchor="start">34.1%</text><text class="t-tick" x="162.0" y="89.0" text-anchor="end">Dense, k=1</text><g class="mark"><title>Dense, k=1: 34.4%</title><rect class="s1" x="170.0" y="73.0" width="321.1" height="22.0" rx="3"/></g><text class="t-val" x="497.1" y="89.0" text-anchor="start">34.4%</text><text class="t-tick" x="162.0" y="119.0" text-anchor="end">Dense, k=5</text><g class="mark"><title>Dense, k=5: 38.1%</title><rect class="s1" x="170.0" y="103.0" width="355.6" height="22.0" rx="3"/></g><text class="t-val" x="531.6" y="119.0" text-anchor="start">38.1%</text><text class="t-tick" x="162.0" y="149.0" text-anchor="end">Random few-shot</text><g class="mark"><title>Random few-shot: 32.2%</title><rect class="s1" x="170.0" y="133.0" width="300.5" height="22.0" rx="3"/></g><text class="t-val" x="476.5" y="149.0" text-anchor="start">32.2%</text><text class="t-tick" x="162.0" y="179.0" text-anchor="end">BM25</text><g class="mark"><title>BM25: 37.5%</title><rect class="s1" x="170.0" y="163.0" width="350.0" height="22.0" rx="3"/></g><text class="t-val" x="526.0" y="179.0" text-anchor="start">37.5%</text><text class="t-tick t-strong" x="162.0" y="209.0" text-anchor="end">Meta-Harness</text><g class="mark"><title>Meta-Harness: 38.8%</title><rect class="s1" x="170.0" y="193.0" width="362.1" height="22.0" rx="3"/></g><text class="t-val" x="538.1" y="209.0" text-anchor="start">38.8%</text></svg><figcaption>The average column of Table 6, from the paper. Random few-shot falls below no retrieval. BM25 and five dense examples both help. The discovered harness has the highest average, 38.8%, 1.3 points above plain BM25.</figcaption></figure>

> [!PAPER] · §4.2
> [![The Results paragraph of Section 4.2, with same BM25-based lexical retrieval stack as the sparse baseline, average gain of 4.7 points, outperforming BM25 retrieval by 1.3 and avoiding the regressions highlighted](/img/papers/meta-harness/p4-results.png)](/img/papers/meta-harness/p4-results.png)
>
> **What it says:** Meta-Harness works "on top of the same BM25-based lexical retrieval stack as the sparse baseline". It beats no retrieval on all five models, "outperforming BM25 retrieval by 1.3 points overall", while "avoiding the regressions" of dense retrieval and random few-shot.
>
> **What it means:**
>
> This is the fair comparison. BM25 Retrieval and Meta-Harness use the same index and the same search formula. The only difference is the policy around it: the router, the dedup, the rerank and the number of examples. So the 1.3 points are the value of the policy alone.
>
> The second claim is about safety. A method that helps on average but hurts two models out of five is risky to deploy. You do not know in advance which kind your model is. The discovered policy helped all five, including four it never saw during the search.

### A tiny version of Table 6 on a laptop

We cannot run GPT-5.4 or Gemini, and IMO problems are far beyond a small local model. So we built the smallest honest copy of the experiment. The frozen model is Qwen2.5-1.5B-Instruct, with greedy decoding. The problems are 24 clean MATH test problems of levels 1 to 3, six from each route. Four harnesses fill the prompt: no retrieval, three random examples, BM25 top 3, and our routed harness.

The harness decides only what goes above the problem:

```python
def build_prompt(problem, examples):
    """The harness's only job here: decide which solved examples go above the problem."""
    parts = []
    if examples:
        parts.append('Here are some solved problems that may help.\n')
        for ex in examples:
            parts.append(f'Problem: {ex["problem"]}\nSolution: {ex["solution"][:SOL_CHARS]}\n')
        parts.append('Now solve this problem.\n')
    parts.append(f'Problem: {problem}')
    return '\n'.join(parts)
```

```text
24 decontaminated MATH test problems, levels 1-3: {'default': 6, 'number_theory': 6, 'combinatorics': 6, 'geometry': 6}
model Qwen/Qwen2.5-1.5B-Instruct on mps, greedy, max 512 new tokens, solutions cut at 1200 chars
    no retrieval: 14/24 =  58.3%   combin 3/6  defaul 3/6  geomet 4/6  number 4/6   prompt    91 tokens    234 s
   (no \boxed answer in 3 outputs)
 random few-shot: 11/24 =  45.8%   combin 2/6  defaul 4/6  geomet 3/6  number 2/6   prompt  1036 tokens    299 s
      BM25 top-3: 12/24 =  50.0%   combin 2/6  defaul 3/6  geomet 4/6  number 3/6   prompt   849 tokens    276 s
  routed harness: 13/24 =  54.2%   combin 2/6  defaul 3/6  geomet 4/6  number 4/6   prompt   923 tokens    181 s
random few-shot vs no retrieval: fixes 1, breaks 4
BM25 top-3 vs no retrieval: fixes 1, breaks 3
routed harness vs no retrieval: fixes 1, breaks 2
```

[![Terminal output of mh_part4_llm.py with the four harnesses](/img/papers/meta-harness/part4_llm-run.png)](/img/papers/meta-harness/part4_llm-run.png)

<figure class="fig"><svg viewBox="0 0 680 180" role="img" aria-label="Bar chart of our accuracy with four harnesses: no retrieval 58.3 percent, random few-shot 45.8 percent, BM25 top-3 50.0 percent, routed harness 54.2 percent."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="30.0" text-anchor="start">Our run: Qwen2.5-1.5B-Instruct, 24 MATH test problems</text><text class="t-tick" x="162.0" y="59.0" text-anchor="end">no retrieval</text><g class="mark"><title>no retrieval: 58.3%</title><rect class="s1" x="170.0" y="43.0" width="233.3" height="22.0" rx="3"/></g><text class="t-val" x="409.3" y="59.0" text-anchor="start">58.3%</text><text class="t-tick" x="162.0" y="89.0" text-anchor="end">random few-shot</text><g class="mark"><title>random few-shot: 45.8%</title><rect class="s1" x="170.0" y="73.0" width="183.3" height="22.0" rx="3"/></g><text class="t-val" x="359.3" y="89.0" text-anchor="start">45.8%</text><text class="t-tick" x="162.0" y="119.0" text-anchor="end">BM25 top-3</text><g class="mark"><title>BM25 top-3: 50.0%</title><rect class="s1" x="170.0" y="103.0" width="200.0" height="22.0" rx="3"/></g><text class="t-val" x="376.0" y="119.0" text-anchor="start">50.0%</text><text class="t-tick" x="162.0" y="149.0" text-anchor="end">routed harness</text><g class="mark"><title>routed harness: 54.2%</title><rect class="s1" x="170.0" y="133.0" width="216.7" height="22.0" rx="3"/></g><text class="t-val" x="392.7" y="149.0" text-anchor="start">54.2%</text></svg><figcaption>Our tiny copy of Table 6: accuracy of Qwen2.5-1.5B-Instruct on 24 clean MATH test problems with four harnesses. With so few problems, one problem is 4.2 points, so small differences are noise.</figcaption></figure>

On our laptop, no harness beat the bare model. No retrieval solved 14 of 24 problems. Random examples solved 11, BM25 top 3 solved 12, and our routed harness solved 13. The last three lines count single problems. The routed harness fixed 1 problem and broke 2. BM25 fixed 1 and broke 3, and random examples fixed 1 and broke 4.

Two lessons hide in these small numbers. The first agrees with the paper: among the three ways to retrieve, the order is the same as in Table 6. Random examples hurt most, plain BM25 hurts less, and the routed policy hurts least. The second is a warning. A 1.5-billion-parameter model reads about 900 tokens of other solutions, and that distracts it more than it helps. The paper's models are much larger, and they can use a good example without being thrown off by it.

With 24 problems, one problem is 4.2 points, and a gap of one or two problems is within noise. We do not claim that our routed harness is better than BM25. We only show the machinery, and that a weak model can lose from retrieval. This is why the paper tests on five models. A policy is only useful if it does not hurt the model you happen to use.

## TerminalBench-2: a harness for an agent {§4.3}

The second experiment of this part is very different. The model no longer answers one question. It works as an **agent**. It types commands into a terminal, reads what comes back, and decides what to do next. It repeats this for many steps, until it thinks the task is done.

> [!DEFINITION] Agent and terminal
> A **terminal** is the text window where a person types commands to a computer (`ls` lists files, `python3 run.py` runs a program). An **agent** here is a language model in a loop: it reads the screen, writes the next command, and repeats. The code that runs this loop is the agent's harness.

> [!PAPER] · §4.3
> [![The first paragraph of Section 4.3, with 89 challenging tasks, fully autonomous execution, Terminus 2 and Terminus-KIRA, search and final evaluation on the same 89-task benchmark, discovery problem, and regex-based audits for task-specific string leakage highlighted](/img/papers/meta-harness/p4-tb2-setup.png)](/img/papers/meta-harness/p4-tb2-setup.png)
>
> **What it says:** TerminalBench-2 has "89 challenging tasks" that need "fully autonomous execution". The search starts from two open harnesses, "Terminus 2 [33] and Terminus-KIRA [25]". The authors do "search and final evaluation on the same 89-task benchmark", and treat it as a "discovery problem". They check for overfitting with manual inspection and "regex-based audits for task-specific string leakage".
>
> **What it means:**
>
> Each task drops the agent into a fresh sandbox with one written instruction. Task names in the paper include `configure-git-webserver`, `protein-assembly` and `path-tracing`. No human answers questions. At the end, an automatic test checks the result, and the task passes or fails.
>
> The second half of the paragraph breaks the rule of the math experiment, and the authors say so. There is no separate test set: the search and the final score use the same 89 tasks. They call this a discovery problem: the goal is to find the best harness for this benchmark, the way a team climbs a public leaderboard. Their reason is cost and size. Each evaluation is expensive, and splitting 89 tasks would leave too few to steer the search.
>
> The price is that the score may overfit. A harness could, in principle, contain hints for specific tasks. So the authors search the evolved code for strings from the tasks, and read it by hand. Keep this in mind when you read Table 7: it is a search score, not a held-out score.

> [!DEFINITION] Sandbox
> A closed, throw-away computer (usually a container) where an agent can run any command without harming anything else. Each TerminalBench-2 task starts in its own fresh sandbox.

> [!DEFINITION] TerminalBench-2 and pass rate
> TerminalBench-2 is a public benchmark of 89 terminal tasks with automatic tests. The **pass rate** is the share of task attempts whose tests pass. The released harness reports each task over 5 attempts.

> [!PAPER] · §4.3, Table 7
> [![The Results paragraph of Section 4.3 beside Table 7, with 76.4% pass rate, Terminus-KIRA 74.7%, unable to reproduce, 37.6% and Goose 35.5% highlighted. The table lists pass rates for Opus 4.6 harnesses from 58.0 to 81.8 and Meta-Harness 76.4, and for Haiku 4.5 harnesses from 13.9 to 35.5 and Meta-Harness 37.6](/img/papers/meta-harness/p4-tb2-results.png)](/img/papers/meta-harness/p4-tb2-results.png)
>
> **What it says:** with Opus 4.6, the discovered harness reaches a "76.4% pass rate", above "Terminus-KIRA (74.7%)", and second on the leaderboard. Only ForgeCode (81.8%) is higher, but the authors were "unable to reproduce" it from its public code. With Haiku 4.5 the discovered harness reaches "37.6%", above the best reported agent ("Goose, 35.5%").
>
> **What it means:**
>
> Table 7 compares harnesses around the same model. Every row in the top half uses Opus 4.6, a strong model. Every row in the bottom half uses Haiku 4.5, a smaller and cheaper one. The column "Auto" marks the one harness that was written by a search, not by people.
>
> On Opus 4.6, the gain over the starting harness is small: 76.4% against 74.7%, which is 1.7 points. On Haiku 4.5 the gain over Terminus-KIRA is larger: 37.6% against 33.7%, which is 3.9 points. A likely reason: a weaker model wastes more steps, so a harness that saves steps helps it more.
>
> The ForgeCode sentence is worth noticing. A leaderboard score is only as good as the code behind it. The authors could not reach 81.8% with the public ForgeCode code, so that number may depend on parts that are not public. Meta-Harness, by contrast, released its full harness, and we read it below.

<figure class="fig"><svg viewBox="0 0 680 270" role="img" aria-label="Bar chart of the Haiku 4.5 rows of Table 7: the hand-written harnesses score 13.9 to 35.5 percent, Meta-Harness 37.6 percent."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="30.0" text-anchor="start">Table 7: TerminalBench-2 pass rate with Haiku 4.5</text><text class="t-tick" x="162.0" y="59.0" text-anchor="end">OpenHands</text><g class="mark"><title>OpenHands: 13.9%</title><rect class="s1" x="170.0" y="43.0" width="129.7" height="22.0" rx="3"/></g><text class="t-val" x="305.7" y="59.0" text-anchor="start">13.9%</text><text class="t-tick" x="162.0" y="89.0" text-anchor="end">Anthropic's agent</text><g class="mark"><title>Anthropic's agent: 27.5%</title><rect class="s1" x="170.0" y="73.0" width="256.7" height="22.0" rx="3"/></g><text class="t-val" x="432.7" y="89.0" text-anchor="start">27.5%</text><text class="t-tick" x="162.0" y="119.0" text-anchor="end">Terminus 2</text><g class="mark"><title>Terminus 2: 28.3%</title><rect class="s1" x="170.0" y="103.0" width="264.1" height="22.0" rx="3"/></g><text class="t-val" x="440.1" y="119.0" text-anchor="start">28.3%</text><text class="t-tick" x="162.0" y="149.0" text-anchor="end">Mini-SWE-Agent</text><g class="mark"><title>Mini-SWE-Agent: 29.8%</title><rect class="s1" x="170.0" y="133.0" width="278.1" height="22.0" rx="3"/></g><text class="t-val" x="454.1" y="149.0" text-anchor="start">29.8%</text><text class="t-tick" x="162.0" y="179.0" text-anchor="end">Terminus-KIRA</text><g class="mark"><title>Terminus-KIRA: 33.7%</title><rect class="s1" x="170.0" y="163.0" width="314.5" height="22.0" rx="3"/></g><text class="t-val" x="490.5" y="179.0" text-anchor="start">33.7%</text><text class="t-tick" x="162.0" y="209.0" text-anchor="end">Goose</text><g class="mark"><title>Goose: 35.5%</title><rect class="s1" x="170.0" y="193.0" width="331.3" height="22.0" rx="3"/></g><text class="t-val" x="507.3" y="209.0" text-anchor="start">35.5%</text><text class="t-tick t-strong" x="162.0" y="239.0" text-anchor="end">Meta-Harness</text><g class="mark"><title>Meta-Harness: 37.6%</title><rect class="s1" x="170.0" y="223.0" width="350.9" height="22.0" rx="3"/></g><text class="t-val" x="526.9" y="239.0" text-anchor="start">37.6%</text></svg><figcaption>The Haiku 4.5 rows of Table 7, from the paper. Hand-written harnesses around the same model range from 13.9% to 35.5%. The harness that Meta-Harness found reaches 37.6%. The row "Anthropic's agent" is Anthropic's own coding agent, the same tool the paper uses as its proposer.</figcaption></figure>

> [!PAPER] · §4.3
> [![The paragraph on the qualitative behavior of the proposer, with both candidates regressed, confounded by the shared prompt intervention and safer additive modification highlighted](/img/papers/meta-harness/p4-tb2-proposer.png)](/img/papers/meta-harness/p4-tb2-proposer.png)
>
> **What it says:** in early iterations, the proposer combined structural fixes with prompt edits, and "both candidates regressed". It then guessed that the regressions were "confounded by the shared prompt intervention", tested the fixes without the prompt change, and turned to a "safer additive modification" that became the best candidate.
>
> **What it means:**
>
> This is a story about reasoning, not luck. Two changes made the harness worse. Both changes had something in common: a rewrite of the prompt. So the proposer asked: was it the fix, or was it the shared prompt rewrite? It tested the fixes alone and saw a much smaller drop. That pointed at the prompt.
>
> After more failed attempts to repair the loop itself, the proposer changed its plan. It stopped editing what the agent does, and only added information at the start. [Part 5](part-5-limits-and-what-came-after.md) walks through the whole trajectory, iteration by iteration. Here we look at the change that won.

> [!PAPER] · Appendix A.2
> [![The iteration 7 passage of Appendix A.2, with purely additive, environment snapshot via a single shell command before the first LLM call, and without risking regression on already-passing tasks highlighted](/img/papers/meta-harness/p4-a2-iter7.png)](/img/papers/meta-harness/p4-a2-iter7.png)
>
> **What it says:** after six regressions from the 64.4% baseline, the proposer wrote that its new candidate is "purely additive". It gathers an "environment snapshot via a single shell command before the first LLM call" and adds it to the first prompt. It should save wasted turns "without risking regression on already-passing tasks".
>
> **What it means:**
>
> The quoted text is the proposer's own note, saved in the search log. Its logic is the logic of a careful engineer. Every earlier change touched code that passing tasks also depend on, so every change could break something that worked. A new block of text at the start of the prompt cannot break the loop. At worst it is ignored.
>
> Note the number 64.4%. That is the score of Terminus-KIRA inside this search run, not the 74.7% of the leaderboard. The paper does not explain the gap. It does show that scores inside a search can differ from final scores. That is one more reason the proposer needed several tries to see a pattern.

## The discovered agent harness {Appendix B.3, Figure 9}

> [!PAPER] · Appendix B.3
> [![The first paragraph of Appendix B.3 and the proposer's recorded hypothesis, with native tool calling, 30KB output cap, multi-perspective completion checklist, environment bootstrapping, and reduce wasted exploration episodes by 3 to 5 turns highlighted](/img/papers/meta-harness/p4-b3.png)](/img/papers/meta-harness/p4-b3.png)
>
> **What it says:** the harness builds on Terminus-KIRA and keeps its "native tool calling", its "30KB output cap" and its "multi-perspective completion checklist". The new part is "environment bootstrapping": before the loop starts, one compound shell command takes a snapshot of the sandbox, and the snapshot goes into the first prompt. The proposer expected it to "reduce wasted exploration episodes by 3--5 turns".
>
> **What it means:**
>
> Picture a plumber called to a house they have never seen. A good one does not start by guessing. They first look: where is the main valve, what kind of pipes, which tools are in the van. Agents without the snapshot do the same, but they do it one command per turn: `pwd`, then `ls`, then `which python3`. Each turn costs a full model call.
>
> The bootstrap does all that looking in one shot, before the model's first turn, and hands it over as text. The model starts its first turn already knowing what is installed.
>
> The three inherited parts are worth a sentence each. Native tool calling means the model returns structured commands through the API's tool feature, not free text that must be parsed. The output cap cuts any terminal output to 30 KB, so one noisy command cannot flood the context. The checklist makes the agent confirm twice before it ends the task; we replay it in code below.

> [!DEFINITION] Tool calling
> A feature of model APIs: the program lists tools (with names and typed arguments), and the model answers with a structured request to call one. The released harness has three tools: `execute_commands`, `task_complete` and `image_read`.

> [!PAPER] · Appendix B.3
> [![The paragraph on what the snapshot includes, with guarded by a 15-second timeout and fails silently, and roughly 80 lines on top of Terminus-KIRA highlighted](/img/papers/meta-harness/p4-b3-snapshot.png)](/img/papers/meta-harness/p4-b3-snapshot.png)
>
> **What it says:** the snapshot has the working directory, a listing of `/app` (at most 20 entries), the versions of Python, GCC, G++, Node, Java, Rust and Go, the package managers, and the memory. The command is "guarded by a 15-second timeout and fails silently". The whole change is "roughly 80 lines on top of Terminus-KIRA".
>
> **What it means:**
>
> "Fails silently" is a design choice. If anything goes wrong, the harness adds nothing and the agent runs as before. So the new code can only add help; it cannot crash a task. That is the "purely additive" promise of the proposer, written into the code.
>
> Eighty lines is a small change for a gain of 1.7 points on a hard public benchmark. The search did not invent a new agent. It found one missing piece of an existing one.

> [!PAPER] · Figure 9
> [![Figure 9 of the paper: a vertical flow from Task instruction to Env bootstrap (pwd, files, languages, package managers, memory, drawn in red) to Initial prompt (task plus snapshot) to Agent loop (native tool calling, 30KB output cap) to Multi-perspective completion checklist, which goes back to the loop on fail and to Task complete on pass](/img/papers/meta-harness/p4-figure9.png)](/img/papers/meta-harness/p4-figure9.png)
>
> **What it says:** the harness "inherits Terminus-KIRA's native tool calling, output cap, and completion checklist (green)". The "environment bootstrap (red)" is the part Meta-Harness found; it gathers a snapshot before the loop, "eliminating early exploratory turns".
>
> **What it means:**
>
> Compare this picture with Figure 8. The math harness is a branching program that runs once per problem. The agent harness is a loop that may run for dozens of turns, with a check at the end that can send it back. The red box sits before the loop, so it runs once, and it never changes the loop itself.

### The bootstrap in code

The released harness is public: one file, `agent.py`, of 1,321 lines. It needs the Harbor framework and an API key to run, so we cannot run the agent. But the bootstrap is a shell command, and we can run that on a laptop. We copied the command from `agent.py` and changed only the folder name, so we can point it at a test folder.

```python
# The command of AgentHarness._gather_env_snapshot, copied from agent.py. Only the folder name is a parameter.
BOOTSTRAP = (
    "echo '@@PWD@@' && pwd && "
    "echo '@@LS@@' && ls -la {app}/ 2>/dev/null && "
    "echo '@@LANG@@' && "
    "(python3 --version 2>&1 || echo 'python3: not found') && "
    "(gcc --version 2>&1 | head -1 || echo 'gcc: not found') && "
    "(g++ --version 2>&1 | head -1 || echo 'g++: not found') && "
    "(node --version 2>&1 || echo 'node: not found') && "
    "(java -version 2>&1 | head -1 || echo 'java: not found') && "
    "(rustc --version 2>&1 || echo 'rustc: not found') && "
    "(go version 2>&1 || echo 'go: not found') && "
    "echo '@@PKG@@' && "
    "(pip3 --version 2>&1 || echo 'pip3: not found') && "
    "(pip --version 2>&1 || echo 'pip: not found') && "
    "(apt-get --version 2>&1 | head -1 || echo 'apt-get: not found') && "
    "echo '@@MEM@@' && free -h 2>/dev/null | head -2 || true"
)
```

The `@@PWD@@`, `@@LS@@` lines are markers. They split one long output into named sections. Method 1 is the parser of `agent.py`, rewritten in a few lines. A marker line opens a new section, and every other line goes into the open one. Method 2 does the same with one regular expression. The script checks that both give the same sections.

```python
def parse_sections(stdout):
    """Method 1: the parser of agent.py. A line @@NAME@@ starts a new section; other lines go to the open one."""
    sections, key, lines = {}, None, []
    for line in stdout.split('\n'):
        if line.startswith('@@') and line.endswith('@@'):
            if key:
                sections[key] = '\n'.join(lines)
            key, lines = line.strip('@'), []
        else:
            lines.append(line)
    if key:
        sections[key] = '\n'.join(lines)
    return sections


def parse_sections_re(stdout):
    """Method 2: one regular-expression split on the marker lines gives [before, NAME, body, NAME, body, ...]."""
    parts = re.split(r'^@@(\w+)@@$', stdout, flags=re.M)
    return {k: v.strip('\n') for k, v in zip(parts[1::2], parts[2::2])}
```

We run it twice. The first run uses a folder that holds three files, as a task folder would. The second uses a folder that does not exist.

```text
== 1. The environment bootstrap, run on this laptop
sections found: ['PWD', 'LS', 'LANG', 'PKG', 'MEM']
Method 1 and Method 2 give the same sections: True
[Environment Snapshot]
Working directory: /tmp/demo
/app contents (6 lines):
total 24
drwxr-xr-x@ 5 admin  staff  160 18 Oct 10:27 .
drwx------@ 3 admin  staff   96 18 Oct 10:27 ..
-rw-r--r--@ 1 admin  staff    2 18 Oct 10:27 data.csv
-rw-r--r--@ 1 admin  staff    2 18 Oct 10:27 main.py
-rw-r--r--@ 1 admin  staff    2 18 Oct 10:27 README.md
Available languages/tools: Python 3.14.6; Apple clang version 21.0.0 (clang-2100.1.1.101); Apple clang version 21.0.0 (clang-2100.1.1.101); v24.21.0; The operation couldn’t be completed. Unable to locate a Java Runtime.; rustc 1.96.1 (31fca3adb 2026-06-26); /bin/sh: go: command not found; go: not found
Package managers: pip 26.1.2 from /opt/homebrew/lib/python3.14/site-packages/pip (python 3.14); /bin/sh: pip: command not found; pip: not found; /bin/sh: apt-get: command not found
snapshot length: 827 characters

-- the same command when the folder does not exist
sections found: ['PWD', 'LS']
[Environment Snapshot]
Working directory: /tmp/demo
/app contents: (none)
```

The first run is what the model would read before its first turn: where it is, which files are there, and what it can use. On this laptop, Python, Node and Rust are present; Go and Java are not. An agent that reads this will not waste a turn trying `go build`.

The second run shows something we did not expect. The parts of the command are joined with `&&`, which means "run the next part only if this one succeeded". When `ls` fails because the folder is missing, every later part is skipped: no languages, no package managers, no memory. The `|| true` at the very end only keeps the whole command from failing. The command assumes that the task files are in `/app`, as in TerminalBench-2, so this does not happen there. But it is a good example of a harness tuned to its benchmark: the same code in another setting would quietly report much less.

### The completion check, replayed

The checklist of Figure 9 has a small state machine. When the agent calls `task_complete` for the first time, the harness does not end the task. It shows the checklist ("Does your solution meet the requirements ... from the perspectives of a test engineer, a QA engineer, and the user") and sets a flag. Only a second `task_complete` right after ends the task. Any command in between clears the flag.

Appendix A.2 tells us that in iteration 4, the proposer wanted to delete the line that clears the flag. It saw agents that had solved a task but then ran a check command and lost the flag. They saw the checklist again and went round in circles for 30 to 60 steps. We replay both rules on the same sequence of actions:

```python
def replay(actions, reset_on_command=True):
    """actions: 'cmd' (run commands) or 'done' (call task_complete). Returns the log and the step it ended on.
    agent.py: the first 'done' shows a checklist and sets pending; a second 'done' right after ends the task;
    any command in between sets pending back to False (the line that iteration 4 wanted to remove)."""
    pending, log = False, []
    for step, a in enumerate(actions, 1):
        if a == 'done':
            if pending:
                log.append(f'{step}:done -> END')
                return log, step
            pending = True
            log.append(f'{step}:done -> checklist')
        else:
            if reset_on_command:
                pending = False
            log.append(f'{step}:cmd')
    return log, None
```

```text
== 2. The completion check: task_complete must be called twice in a row
agent.py (reset on command): 1:cmd 2:cmd 3:done -> checklist 4:cmd 5:done -> checklist 6:cmd 7:done -> checklist 8:done -> END
   checklists shown: 3, task ends at step 8
iteration 4 idea (no reset): 1:cmd 2:cmd 3:done -> checklist 4:cmd 5:done -> END
   checklists shown: 1, task ends at step 5
```

With the rule of `agent.py`, an agent that checks its work after each checklist sees the checklist three times and ends at step 8. Without the reset, it ends at step 5, which looks like a clear win for the iteration 4 idea. But the search said otherwise: that candidate scored lower than the baseline.

Our reading is that the reset has a purpose. It makes the agent confirm right after its last change, not after some earlier state. Our replay shows the cost of the rule, and only the benchmark could show its value. The released harness keeps the reset.

[![Terminal output of mh_part4_tb2.py: the snapshot on a laptop and the replay of the completion check](/img/papers/meta-harness/part4_tb2-run.png)](/img/papers/meta-harness/part4_tb2-run.png)

> [!PAPER] · Appendix B.3
> [![The per-task analysis, with gains on 7 of 89 tasks and domain-specific tooling whose availability cannot be assumed in advance highlighted](/img/papers/meta-harness/p4-pertask-1.png)](/img/papers/meta-harness/p4-pertask-1.png)
> [![The end of the per-task analysis on the next page, with those wasted turns can be the difference between pass and fail highlighted](/img/papers/meta-harness/p4-pertask-2.png)](/img/papers/meta-harness/p4-pertask-2.png)
>
> **What it says:** compared with Terminus-KIRA, the harness "gains on 7 of 89 tasks", most on `protein-assembly` and `path-tracing`. These tasks need "domain-specific tooling whose availability cannot be assumed in advance": bioinformatics libraries, rendering tools, chess engines. On tasks with tight turn budgets, "those wasted turns can be the difference between pass and fail".
>
> **What it means:**
>
> The gain is not spread evenly. On 82 of the 89 tasks, the snapshot changes nothing. On 7 tasks it flips the result. Those are the tasks where the first question is "what is even installed here?", and a wrong guess sends the agent down a long dead end.
>
> This also explains why the bootstrap helps Haiku 4.5 more than Opus 4.6. A strong model recovers from a wrong first guess within a turn or two. A weaker model is more likely to keep going down the wrong path until it runs out of turns.

## What our small runs cannot show

Our math experiment has the shape of the paper's, but almost every number is smaller. Our corpus has 7,500 problems, not 535,356, and our questions are school competition problems, not IMO problems. Our model has 1.5 billion parameters, and the paper's models are far larger. Our policy was written by us from the paper's description, while the paper's was selected from 109 candidates by a 40-iteration search. Our results can show how each piece works. They cannot confirm or refute the 4.7-point gain.

For TerminalBench-2 we ran no agent at all. Each run needs paid API calls to Opus 4.6 or Haiku 4.5, and each task takes minutes in its own container. What we could do is read the released code and run its two key pieces as plain Python. The snapshot command and the completion rule are the same code that scored 76.4%; only the model and the sandbox are missing.

> [!TAKEAWAYS] Key takeaways
> - For math, Meta-Harness searched over a **retrieval policy**: which solved problems to put in the prompt, how many, and in what order. The corpus has 535,356 solved problems, decontaminated with prefix matches and word Jaccard at 0.8.
> - The search ran 40 iterations on 250 search problems with one model and made 109 candidates. One harness was then tested on 200 unseen IMO-level problems and four models it never saw.
> - The discovered harness is a lexical router plus four BM25 recipes. Combinatorics: dedup and rerank. Geometry: one hard reference plus two raw matches. Number theory: a technique bonus. The rest: an adaptive number of examples.
> - It averaged 38.8% against 34.1% with no retrieval (+4.7), and 37.5% for plain BM25 (+1.3). It improved all five models; dense retrieval and random examples hurt some of them.
> - BM25 scores rare words higher, lets extra hits saturate, and favours short documents. Our from-scratch BM25 matches `rank_bm25` to $$10^{-16}$$ on a hand example and $$3.6 \times 10^{-12}$$ on MATH.
> - On MATH, 491 of 5,000 test problems had a near copy in the training half. Decontamination is not optional.
> - Our keyword router got 77.6% of clean MATH test problems to the right route. All three of our retrievers pick examples of the same subject about 62% of the time. The policy changes which neighbours, not where they come from.
> - On TerminalBench-2 there is no held-out set; search and score use the same 89 tasks. The discovered harness reached 76.4% with Opus 4.6 (Terminus-KIRA: 74.7%) and 37.6% with Haiku 4.5, first among Haiku 4.5 agents.
> - The winning change is about 80 lines: one shell command that takes a snapshot of the sandbox before the first turn. It changed the result on 7 of 89 tasks, all of them tasks that need unusual tools.
> - The proposer found it after six regressions, by reasoning that changes to the prompt and the loop were risky and an additive change was not.

## The paper and the code

[Paper (arXiv 2603.28052)](https://arxiv.org/abs/2603.28052) · [Project page](https://yoonholee.com/meta-harness/) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness)
