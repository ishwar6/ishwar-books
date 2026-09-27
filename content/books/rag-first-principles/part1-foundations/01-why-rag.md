# Chapter 1 · Why RAG Exists

> **Goal:** by the end of this chapter you can explain, without notes, what problem
> retrieval-augmented generation solves, why the alternatives (fine-tuning, giant context
> windows) do not make it obsolete, and what every box in a RAG pipeline is for. You will
> also have the vocabulary the rest of this book uses. No code in this chapter: it is the
> only one.

---

## 1.1 Four things a language model cannot do on its own

A large language model (LLM) is a function from text to text. Everything it "knows" was
baked into its weights during training. That has four consequences, and RAG exists because
of them.

**1. It stops knowing at the cutoff date.** A model trained until spring 2026 has no idea
what your company shipped last Tuesday. You cannot ask it about the Beacon 4.2 release
notes because those did not exist when it was trained.

**2. It has never seen your private data.** Your handbook, your tickets, your contracts,
your database. None of it was on the public internet, so none of it is in the weights. Ask
"how many PTO days do I get" and the model can only answer for *some* company, not yours.

**3. It makes things up: fluently.** When the model does not know, it does not say "I
don't know". It produces the most *plausible-sounding* continuation. This is called
**hallucination**, and the dangerous part is that a hallucinated answer reads exactly like
a correct one. Ask for a policy that does not exist and you will get one, with numbers.

**4. Its input is finite and costs money.** The **context window** is the maximum number
of tokens the model can read in one call. Even models with million-token windows charge per
token, get slower as the input grows, and (this is measured, not folklore) get *worse*
at using facts buried in the middle of a very long input ("lost in the middle"). Pasting
your entire wiki into every request is neither free nor reliable.

RAG attacks all four at once with one move: **look the facts up at question time and put
them in the prompt.** The model no longer has to *remember* your handbook; it has to
*read* the three paragraphs you hand it and answer from those.

## 1.2 The three ways to give a model new knowledge

| | Fine-tuning | Long context ("just paste everything") | RAG |
|---|---|---|---|
| How | continue training the model on your documents | put all documents in every prompt | retrieve the relevant few pieces per question, put only those in the prompt |
| Updates | retrain (hours–days, $$$) | instant | instant: re-index the changed document |
| Cost per question | low (short prompt) | very high (whole corpus every time) | low–medium (a few chunks) |
| Scales to | whatever fits in training | ~1M tokens ≈ 750k words ≈ a couple of thousand pages | terabytes: retrieval is the filter |
| Provenance ("where did that come from?") | none: it is in the weights | possible but the model must find it | built in: you know exactly which chunks were shown |
| Access control | impossible per user | possible but wasteful | natural: filter retrieval by the user's permissions |
| Hallucination | still happens; fine-tuning teaches *style* far better than *facts* | reduced, if the model finds the fact | reduced, and *checkable* against the retrieved text |
| Good for | tone, format, a new skill, a domain's jargon | small, stable corpora; one-off analysis of a document | facts that change, private data, large corpora, anything that needs citations |

The honest summary: **fine-tuning changes how a model behaves; RAG changes what it knows
right now.** They are not rivals: production systems often do both (fine-tune a small
model to follow your answer format, RAG to feed it facts). Long context is a genuine
competitor for *small* corpora and a genuine complement inside RAG (bigger context lets
you retrieve more generously). Chapter 19 returns to the "is RAG dead?" debate with 2026
evidence; the short answer is that retrieval got *more* important as agents started reading
more, not less.

## 1.3 The RAG loop

Two independent pipelines. The first runs when documents change; the second runs when a
user asks something.

```
INGEST (write path - runs when data changes)

  documents ──► parse ──► chunk ──► embed ──► store
  (PDF, md,     (text +   (pieces   (vector    (vector DB:
   HTML, DB)     metadata) ~200–800  per chunk) vector + text
                           tokens)              + metadata)

QUERY (read path - runs per question)

  question ──► embed ──► search ──► (rerank) ──► top-k chunks ──► prompt ──► LLM ──► answer
              (same      (nearest   (optional:   (3–10 pieces    (system     (reads   (+ citations)
               model!)    vectors,   a stronger   of text)        rules +     the
                          + filters) model         	             context +   context,
                                     re-orders)                   question)  answers)
```

Every chapter of this book is about one box in this diagram, or about measuring whether a
box is doing its job:

| Box | What goes wrong there | Chapter |
|---|---|---|
| parse | tables destroyed, headers lost, PDFs read in the wrong order | 5, 7 |
| chunk | answer split across two chunks; chunk too big to be specific | 5 |
| embed | wrong model, docs and queries embedded differently, cost | 2 |
| store | can't filter by metadata, too slow at scale, no persistence | 4, 18 |
| search | semantic search misses exact terms (IDs, codes, names) | 8 |
| rerank / top-k | right chunk retrieved at rank 9, k was 5 | 9 |
| prompt | model ignores the context, or answers from memory | 11 |
| LLM | hallucinates, refuses, invents citations | 11, 12 |
| the whole loop | you *feel* it's better but can't prove it | 10, 13 |
| the loop, but the model decides when to search | agents, loops, budgets | 14–16 |

## 1.4 Vocabulary

You will use these words in every interview about RAG. Learn the *precise* meaning.

- **Token**: the unit models read and bill by. Roughly ¾ of an English word. Common
  words are one token; "Retrieval" is two or three (`Ret`/`rie`/`val` in OpenAI's
  `cl100k_base` tokenizer) and "Lumora" is two or three.
- **Embedding**: a fixed-length list of numbers (a **vector**) produced by an embedding
  model from a piece of text, arranged so that texts with similar *meaning* get vectors
  that point in similar directions. `text-embedding-3-small` gives 1,536 numbers per text.
- **Cosine similarity**: the cosine of the angle between two vectors. 1 = same direction,
  0 = unrelated, −1 = opposite. For normalized vectors (length 1) it equals the dot product.
  This is the number that ranks search results. Chapter 2 computes it by hand.
- **Chunk**: a piece of a document small enough to be one search result and one prompt
  ingredient. Typically 200–800 tokens. Chunking strategy is the single highest-leverage
  decision in a RAG system (Chapter 5).
- **Vector database**: a store that keeps vectors together with their text and metadata
  and answers "which stored vectors are closest to this one?" fast, even for millions of
  vectors. We use **Qdrant**.
- **ANN / HNSW**: *approximate nearest neighbour* search. Comparing a query against every
  stored vector is exact but O(n). **HNSW** (Hierarchical Navigable Small World) is a graph
  index that finds *almost* the same neighbours in roughly O(log n). "Approximate" is the
  price of speed; you tune how approximate (Chapter 4, 18).
- **Top-k**: the k best-scoring chunks the retriever returns. k is a dial: too small and
  you miss the answer; too large and you dilute the context and pay for tokens.
- **Retriever**: the component that turns a question into a list of chunks. Dense
  (vector) search is one retriever; keyword search (BM25) is another; hybrid combines them
  (Chapter 8).
- **Reranker**: a second, slower model that re-scores the top-k candidates with the
  question and each chunk *together*, producing a better order than embedding similarity
  alone (Chapter 9).
- **Context window**: the maximum tokens an LLM can take in one call, prompt plus answer.
- **Grounding**: making the model's answer rest on provided text. A **grounded** answer
  is one whose every claim can be traced to a retrieved chunk. **Faithfulness** is the
  metric for it (Chapter 10).
- **Hallucination**: a fluent, confident, false statement. In RAG it splits into two:
  *the retriever brought the wrong text* (fix retrieval) and *the model ignored or
  embellished the right text* (fix the prompt/model). Diagnosing which is Chapter 11.
- **Golden set**: a list of questions with known correct answers and known source
  documents. Without one you cannot measure anything (Chapter 10). Ours is
  [`data/golden/qa.yaml`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/golden/qa.yaml).
- **Agentic RAG**: instead of "always retrieve once, then answer", the model *decides*
  whether to search, what to search for, whether the results are good enough, and whether
  to search again. Implemented with LangGraph in Part 5.

## 1.5 A transformer, in three sentences, and why it matters here

Interviewers ask this. Here is the answer at the right depth.

A **transformer** processes all tokens of its input at once and, in each layer, lets every
token look at every other token through **self-attention**: a learned weighting of "which
other tokens matter for understanding this one". Stacking these layers turns raw tokens
into contextual representations, which is why "bank" next to "river" ends up represented
differently from "bank" next to "loan". The 2017 paper *Attention Is All You Need*
introduced it, and it displaced recurrent networks because attention is parallel (fast to
train on large data) and has no distance penalty (a token at position 1 can attend to
position 900 directly).

**Why it matters for RAG:** an embedding model *is* a transformer whose output for a text
is pooled into one vector. Because the representation is contextual, "PTO", "annual leave"
and "vacation days" land close together. That is the entire reason semantic search works:
and also the reason it fails on things transformers do not treat as *meaning*, like a
part number or a person's name, which is why Chapter 8 adds keyword search back.

## 1.6 Map of this book

- **Part 1: Foundations (Ch 1–3).** Vocabulary, embeddings by hand, then the smallest
  complete RAG in ~70 lines of LangChain.
- **Part 2: Building (Ch 4–7).** Qdrant, chunking (measured), a reusable pipeline, and
  growing the corpus.
- **Part 3: Retrieval (Ch 8–9).** BM25, hybrid search, reranking, query rewriting,
  metadata filtering.
- **Part 4: Quality (Ch 10–13).** Every metric with its formula and how to move it,
  hallucination, "not enough data", observability.
- **Part 5: Agentic (Ch 14–16).** LangGraph, agentic RAG, multi-agent guardrails
  (loops, budgets, RAG-first enforcement).
- **Part 6: Scale (Ch 17–18).** Serving, security, Docker, and the maths of a terabyte.
- **Part 7: Frontier (Ch 19–20).** What changed in 2025–26, and an interview question bank.
- **Appendices.** Papers in reading order, metrics cheat-sheet, glossary, API cheat-sheet.

Everything runs against one corpus: a fictional company's handbook in [`data/handbook/`](https://github.com/ishwar6/ishwar-books/tree/main/code/rag/data/handbook)
(14 documents) and 47 golden questions in [`data/golden/qa.yaml`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/golden/qa.yaml). The corpus is small on
purpose: you can read all of it in twenty minutes, which means you can *check* every
answer the system gives. That habit is the course.

## Run it

Nothing to run. Read [`data/handbook/02-pto-and-leave-policy.md`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/handbook/02-pto-and-leave-policy.md) and [`data/handbook/14-faq.md`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/handbook/14-faq.md)
now and notice that they disagree about PTO days (24 vs 20). Remember that; it comes back
in Chapters 3, 11 and 12.

```bash
cat data/handbook/02-pto-and-leave-policy.md | head -20
grep -n "PTO" data/handbook/14-faq.md
```

## Exercises

1. Write down, from memory, the two pipelines of the RAG loop with every box. Compare
   with §1.3. Do it again tomorrow.
2. Pick a question you might ask about your *own* employer. For each of the four LLM
   limits in §1.1, say whether it applies to that question.
3. For each row of the comparison table in §1.2, think of one real situation where that
   column is the *right* choice. (Fine-tuning has one; find it.)
4. Read the golden set [`data/golden/qa.yaml`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/data/golden/qa.yaml). Which questions do you predict will be hard
   for a naive system and why? Write the ids down; check your predictions in Chapter 10.

## Interview questions

**Q: Explain RAG end to end.**
Two pipelines. Ingestion: parse documents into text with metadata, split into chunks of a
few hundred tokens, embed each chunk with an embedding model, store vector + text +
metadata in a vector database. Query: embed the user's question with the *same* model,
find the top-k nearest chunks (optionally filtered by metadata and reranked), build a
prompt with a system instruction, the chunks as numbered context and the question, and let
the LLM answer from that context with citations. Evaluate both halves separately:
retrieval with recall@k, generation with faithfulness and correctness.

**Q: Why embeddings? Why not just keyword search?**
Keyword search matches strings; embeddings match meaning, so "annual leave" finds the PTO
policy. They fail in opposite places (embeddings are weak on exact identifiers, keywords
on paraphrase) which is why production systems use both (hybrid search).

**Q: Why store the embeddings? Can't we compute them at query time?**
The corpus side must be embedded once and searched many times; recomputing millions of
vectors per query is impossibly slow and expensive. Storing them in a vector DB with an
ANN index makes each query a millisecond graph walk instead of a full scan. Only the
question is embedded at query time.

**Q: RAG vs fine-tuning: when would you pick which?**
Fine-tune to change *behaviour*: tone, output format, a domain's jargon, a task the base
model does poorly. Use RAG to change *knowledge*: facts that change, private data, anything
that needs a citation or per-user access control. Fine-tuning does not reliably teach
facts and cannot be updated in minutes; RAG cannot teach a new skill. Often you do both.

**Q: What is a transformer, in a few sentences, and why does it matter for embeddings?**
A neural network that processes a whole sequence in parallel and uses self-attention so
every token's representation is shaped by every other token; introduced in "Attention Is
All You Need" (2017); replaced RNNs because it trains in parallel and handles long-range
dependencies. Embedding models are transformers whose contextual token representations are
pooled into one vector: that contextuality is why semantically similar texts end up
nearby.

**Q: Won't million-token context windows kill RAG?**
For a corpus that fits, long context is a real option and a simpler one. It does not fit
for most enterprises (terabytes), it costs the whole corpus in tokens per question, it
cannot enforce per-user access, and models measurably degrade at using facts buried in
very long inputs. In practice long context made retrieval *more* useful: you can afford to
retrieve more, and agents read more per step.

**Q: What is hallucination in a RAG system and what are its two causes?**
A confident false statement. Either retrieval failed (the right text was never shown, so
the model filled the gap) or generation failed (the right text was shown and the model
ignored, misread or embellished it). You diagnose which by looking at the retrieved chunks
first: that is why every script in this book prints them.

## Key takeaways

- RAG exists because LLMs have a cutoff, have not seen your data, hallucinate, and have a
  finite paid-for input. Retrieval fixes all four by putting the right text in the prompt.
- Fine-tuning changes behaviour; RAG changes knowledge. Long context is a complement, not
  a replacement: a million-token window is roughly two thousand pages, and you pay for all
  of them on every question.
- Two pipelines: ingest (parse → chunk → embed → store) and query (embed → search →
  rerank → prompt → LLM). Every RAG bug lives in exactly one box.
- The same embedding model must be used for documents and queries.
- You cannot improve what you do not measure; a golden set comes before optimisation.

## Next

→ [Chapter 2: Embeddings: Meaning as Geometry](02-embeddings.md)

Chapter 2 makes "embedding" concrete: you will embed sentences, compute cosine similarity
by hand, and watch nearest-neighbour search work (and fail) on real vectors.
