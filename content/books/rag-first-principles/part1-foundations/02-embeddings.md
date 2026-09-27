# Chapter 2 · Embeddings: Meaning as Geometry

> **Goal:** you can explain what an embedding is, compute cosine similarity by hand and
> say why it is the same thing as a dot product on normalized vectors, run a
> nearest-neighbour search with numpy, and estimate what embedding a corpus costs. You
> will also see the first two failure modes of semantic search with your own eyes.

Code for this chapter: [`code/ch02/embeddings_playground.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch02/embeddings_playground.py).

---

## 2.1 What an embedding is

Call the embedding model with a sentence; get back a list of floats.

```python
from ragbook import get_embeddings          # wraps langchain_openai.OpenAIEmbeddings(model=...)

embeddings = get_embeddings()
vec = embeddings.embed_query("Full-time employees receive 24 days of PTO per calendar year.")
len(vec)          # 1536 for text-embedding-3-small
vec[:5]           # [-0.0228, 0.0292, 0.0189, 0.0682, -0.0033]
```

That is all it is: **a point in a 1,536-dimensional space**. The model was trained so that
texts that *mean* similar things land close together. Nobody knows what dimension 812
"means"; the geometry is the meaning.

Two LangChain methods, one difference:

- `embed_documents(list[str])`: batch, for the corpus side.
- `embed_query(str)`: single, for the question side. For OpenAI models these produce
  identical vectors for identical text; some other models add a task prefix for queries
  vs documents (asymmetric retrieval: see §2.5), and this split is where they do it.

OpenAI embeddings come back **unit length** (norm ≈ 1.0). Hold that thought.

## 2.2 Three distances, one ranking

`embeddings_playground.py` embeds ten handbook-style sentences, then compares the PTO
sentence with (a) the carry-over sentence (related) and (b) the robot payload sentence:
unrelated:

```
cosine(PTO, carry-over) = 0.382  cosine(PTO, payload) = 0.082
dot   (PTO, carry-over) = 0.382  dot   (PTO, payload) = 0.082
eucl  (PTO, carry-over) = 1.112  eucl  (PTO, payload) = 1.355
```

The formulas, and why the numbers line up:

- **Cosine similarity**: `cos(a, b) = (a · b) / (|a| · |b|)`. The cosine of the angle
  between the vectors. Ignores length; only direction matters. Range −1…1.
- **Dot product**: `a · b = Σ aᵢbᵢ`. If both vectors have length 1, then `|a| · |b| = 1`
  and the dot product *is* the cosine. That is why the two rows are identical.
- **Euclidean distance**: `|a − b|`. For unit vectors, `|a − b|² = 2 − 2·cos(a, b)`.
  Smaller distance ⇔ larger cosine. Same ranking, opposite direction.

```python
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

**Practical consequence.** With normalized embeddings, "which metric should I use?" is a
non-question: they rank identically. Qdrant asks you to pick one when you create a
collection (`Distance.COSINE`, `DOT`, `EUCLID`); pick `COSINE` and move on. The only time
it matters is when vectors are *not* normalized (some models, or when you scale vectors on
purpose to encode popularity or recency into the dot product).

**What the absolute numbers mean.** Nothing universal. 0.38 for "clearly related" and 0.08
for "unrelated" is typical of `text-embedding-3-*`; other models put related pairs at 0.8.
Never copy a similarity threshold from a blog post: measure it on your own model and data
(Chapter 12).

## 2.3 Nearest neighbours: the heart of retrieval

```python
def rank(query_vec, doc_vecs, k=3):
    sims = [cosine(query_vec, d) for d in doc_vecs]
    order = np.argsort(sims)[::-1][:k]
    return [(int(i), sims[i]) for i in order]
```

This eight-line function **is** vector search. A vector database is this loop, made fast
for millions of rows and given filters and persistence (Chapter 4). Everything you learn
about it here transfers.

Four queries, chosen to show what embeddings can and cannot do:

```
Q: 'how much holiday do I get'            <- shares NO words with the PTO sentence
   0.420  Up to 5 unused vacation days can be carried over until 31 March.
   0.406  Full-time employees receive 24 days of PTO per calendar year.
   0.352  Parental leave is 26 weeks for the primary caregiver.

Q: 'how heavy a load can the robot lift'  <- paraphrase
   0.649  The Atlas A2 robot carries a maximum payload of 250 kg.
   0.236  A full charge of the Atlas battery takes 90 minutes.

Q: 'how often do passwords expire'        <- 'expire' vs 'rotate'
   0.689  Privileged account passwords rotate every 90 days.
   0.252  Report security incidents to security@lumora.example within one hour.

Q: '90'                                   <- a bare number
   0.223  Privileged account passwords rotate every 90 days.
   0.180  A full charge of the Atlas battery takes 90 minutes.
```

Read these carefully; two lessons are hiding in them.

**Lesson 1: semantic search works across vocabulary.** "holiday" found "vacation" and
"PTO"; "lift" found "payload"; "expire" found "rotate". No keyword system does this. This
is the whole reason to embed.

**Lesson 2: it is fuzzy where you want it exact.** "how much holiday do I get" ranked the
*carry-over* sentence above the one that actually answers the question (0.420 vs 0.406).
Both are "about vacation days"; the embedding cannot tell which one *answers* the question.
And the bare query `90` scored everything low and nearly tied: a number is not a meaning.
Chapter 8 (keyword search) and Chapter 9 (reranking) exist for exactly these two gaps.

## 2.4 Matryoshka: shorter vectors, almost the same answer

`text-embedding-3-*` models are trained so that the *first* n dimensions are themselves a
good embedding (Matryoshka Representation Learning). Ask for fewer dimensions and you get
a smaller vector, not a different model:

```python
small = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=256)
```

```
Same model with dimensions=256 -> shape (10, 256)
top-3 overlap 1536 vs 256 dims: 11/12 slots agree  (6x less storage)
```

Six times less memory and disk, six times faster brute-force search, and the top-3 sets
agree in 11 of 12 slots. The top hit is identical for all four queries; the one
disagreement is in the weak tail: for "how often do passwords expire" the 256-dim vectors
pick a different low-scoring distractor (≈0.25–0.33) for third place. The 0.420 vs 0.406
near-tie above *did* flip order at 256 dims, which is why the script compares top-3 *sets*
rather than first places. At terabyte scale (Chapter 18) this dial is one of the first you
reach for, together with quantization.

## 2.5 The two rules of embedding hygiene

**Rule 1: the same model embeds documents and queries.** Two different embedding models
produce vectors in *unrelated* spaces; a cosine between them is noise. If you ever change
the embedding model, you re-embed the entire corpus. Write the model name into the
collection's metadata so nobody mixes them (ragbook only pins it in `.env` as
`RAG_EMBED_MODEL`; a production system records it next to the collection).

**Rule 2: know whether your model is symmetric or asymmetric.** A *symmetric* model
expects the query and the document to be the same kind of text (sentence vs sentence). An
*asymmetric* model is trained for short question → long passage and often wants a prefix
("query: …" / "passage: …", or an `input_type` parameter). OpenAI's models need no prefix;
many open-source models (E5, BGE, Nomic) do, and forgetting it costs several points of
recall silently. LangChain's `embed_query` / `embed_documents` split exists so the
integration can add the right prefix for you.

> Choosing *which* embedding model, systematically rather than by vibes or vendor blog, is
> [Chapter 25](../part8-deep-dives/25-choosing-an-embedding-model.md): a measured bake-off across
> API and local models, instruction prefixes, sequence limits and migration.

## 2.6 Tokens and money

```python
enc = tiktoken.get_encoding("cl100k_base")       # the tokenizer used by text-embedding-3-*
total_tokens = sum(len(enc.encode(s)) for s in SENTENCES)
```

```
10 sentences = 132 tokens
  text-embedding-3-small: $0.02/1M tokens -> this corpus $0.000003; 1 GB of text ~ $5
  text-embedding-3-large: $0.13/1M tokens -> this corpus $0.000017; 1 GB of text ~ $32
```

Rules of thumb worth memorising for interviews:

| | |
|---|---|
| 1 token | ≈ 4 characters ≈ ¾ word of English |
| 1 GB of plain text | ≈ 250 M tokens |
| Embedding 1 GB with `-small` | ≈ $5, one-off |
| Embedding 1 GB with `-large` | ≈ $32, one-off |
| Storage for 1 GB of text chunked at 500 tokens, 1536 float32 dims | 500 k chunks × 6 KB ≈ **3 GB of vectors**: the vectors are bigger than the text |

That last row is the surprise. The *text* is cheap; the *vectors* are what you pay to keep
in RAM (Chapter 18). Matryoshka and quantization are how you shrink them.

Embedding models also have a maximum input (8,191 tokens for `text-embedding-3-*`).
LangChain's `OpenAIEmbeddings` silently splits longer inputs and averages the pieces; do
not rely on that: chunk first (Chapter 5).

## Run it

```bash
uv run python code/ch02/embeddings_playground.py
```

Expected (trimmed; your numbers will differ in the third decimal):

```
model=text-embedding-3-small  shape=(10, 1536)  (10 sentences x 1536 floats)
vector norm (OpenAI embeddings come out unit-length): 1.0003

cosine(PTO, carry-over) = 0.382  cosine(PTO, payload) = 0.082
dot   (PTO, carry-over) = 0.382  dot   (PTO, payload) = 0.082
-> on unit vectors: cosine == dot, and euclidean^2 == 2 - 2*cosine. Same ranking, three names.

Nearest neighbours (cosine):
  Q: 'how heavy a load can the robot lift'
     0.649  The Atlas A2 robot carries a maximum payload of 250 kg.
...
top-3 overlap 1536 vs 256 dims: 11/12 slots agree  (6x less storage)
10 sentences = 132 tokens
```

Cost of one run: well under a cent.

## Exercises

1. Add the sentence "Employees get twenty-four days of annual leave" (words, not digits)
   and query with "24 days PTO". Does the embedding connect "twenty-four" to "24"?
2. Embed the same sentence twice. Is the vector bit-identical? (Run it a few times.)
   What does that imply for caching embeddings by text hash?
3. Change `dimensions=256` to `64`. At what size does the top-3 overlap fall apart?
4. Compute the cosine between "The robot is fast" and "The robot is slow". Explain the
   number to someone who expected it to be negative.
5. Using the rules of thumb, estimate the one-off embedding cost and the vector storage
   for a 40 GB document archive with `-small`, chunks of 400 tokens, 1536 dims float32.

## Interview questions

**Q: What is an embedding?**
A fixed-length vector produced by a neural model from a piece of text, arranged so that
semantically similar texts have vectors pointing in similar directions. It turns "how
similar in meaning are these two texts" into a geometric operation: cosine of the angle.

**Q: Cosine similarity vs dot product vs Euclidean distance: which should you use?**
For normalized (unit-length) vectors they rank identically: dot equals cosine, and
Euclidean distance squared equals 2 − 2·cosine. OpenAI embeddings are normalized, so use
cosine and stop worrying. The choice only matters for un-normalized vectors, where the dot
product also rewards vector length.

**Q: Why must documents and queries use the same embedding model?**
Each model defines its own vector space; a vector from model A has no geometric
relationship to one from model B. Mixing them produces meaningless similarities. Changing
the model means re-embedding the whole corpus.

**Q: What does `dimensions=256` do with OpenAI's text-embedding-3 models?**
They are trained with Matryoshka Representation Learning, so the first n dimensions form a
usable embedding on their own. Truncating to 256 cuts storage and search cost ~6× with a
small loss of ranking quality: useful at scale, and a cheap first-stage filter before
re-scoring with the full vector.

**Q: How would you estimate the cost of embedding a corpus?**
Tokens ≈ characters ÷ 4; multiply by price per token (`-small` is $0.02 per million).
1 GB of text ≈ 250 M tokens ≈ $5. The bigger cost is storing vectors: 1536 float32 = 6 KB
per chunk, so vectors typically exceed the text they index by 2–3×.

**Q: Where does semantic search fail?**
Exact identifiers and numbers (part numbers, ticket IDs, "90"), rare proper nouns,
negation, and distinguishing "about the topic" from "answers the question". The fixes are
hybrid search with BM25 and a reranker.

**Q: What is asymmetric retrieval?**
Query and document are different kinds of text (a short question vs a long passage). Some
models are trained for that and expect a prefix or task type on each side; using them
symmetrically or forgetting the prefix degrades recall. LangChain's `embed_query` vs
`embed_documents` exists for this.

## Key takeaways

- An embedding is a point in high-dimensional space; similar meaning ⇒ nearby points.
- Cosine, dot and Euclidean give the same ranking on unit vectors. Use cosine.
- Nearest-neighbour search over vectors *is* retrieval; a vector DB is that loop, made fast.
- Embeddings bridge vocabulary ("holiday" → "PTO") but are fuzzy on exactness (numbers,
  IDs, "which one actually answers").
- Same model on both sides, always. Vectors cost more to store than the text they index.

## Next

→ [Chapter 3: Your First RAG (LangChain Only)](03-first-rag.md)

Chapter 3 assembles these pieces into a working question-answering system over the whole
handbook: in about seventy lines of LangChain.
