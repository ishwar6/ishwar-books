# Chapter 6 · A Generic, Reusable RAG Pipeline

> **Goal:** you have a small codebase (not a script) that you could start a real project
> from: an idempotent ingest, a retriever with filters and thresholds, a generator with
> citations, conflict handling and cost accounting, and a CLI. You understand which parts
> LangChain provides and which parts are yours forever.

Code: [`code/ch06/ingest.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch06/ingest.py), [`code/ch06/retrieve.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch06/retrieve.py), [`code/ch06/generate.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch06/generate.py),
[`code/ch06/rag_cli.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch06/rag_cli.py). The finished pipeline is also packaged as [`ragbook/index.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/index.py),
which every later chapter calls.

---

## 6.1 Four files, two paths

```
ingest.py      write path   load_sources() -> chunk_documents() -> stable ids -> Qdrant "ch06_handbook"
retrieve.py    read path 1  get_store() -> retrieve(question, k, doc_id, score_threshold) -> build_context()
generate.py    read path 2  PROMPT | llm -> answer(question) -> {answer, sources, usage, cost_usd}
rag_cli.py     interface    one question / --interactive / --golden N
```

The split is deliberate. Ingest runs on a schedule or a webhook when documents change.
Retrieval and generation run per request, and are separately testable: Chapter 10
evaluates `retrieve()` with recall metrics and `answer()` with faithfulness metrics, and
they fail independently.

## 6.2 Ingest: loaders, chunks, stable ids

```python
def load_sources() -> list[Document]:
    docs = load_handbook()
    for path in (DATA_DIR / "generated").glob("*.md"):        # Chapter 7 adds these
        docs.append(Document(page_content=path.read_text(), metadata={"source": path.name, "doc_id": path.stem, ...}))
    return docs
```

**Loaders.** `load_handbook()` is fifteen lines of `Path.read_text`. In a real project you
would use LangChain's loaders, which all return the same `list[Document]`:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
DirectoryLoader("data/handbook", glob="**/*.md", loader_cls=TextLoader).load()
PyPDFLoader("report.pdf").load()          # one Document per page, metadata["page"]
```

Whatever the loader, **normalize metadata at the door**: every Document gets `source`
(where it came from), `doc_id` (stable key) and `title`. Everything downstream (filters,
citations, incremental updates) depends on those three fields existing everywhere.

**Chunking** is `ragbook.chunk_documents` (recursive, headings first, 800/120) and adds
two metadata fields per chunk:

```python
meta = dict(doc.metadata, chunk_index=i)
meta["chunk_id"] = str(uuid.uuid5(NAMESPACE, f"{meta['doc_id']}::{i}"))
```

**Stable ids are the whole trick to idempotency.** `uuid5` is a deterministic hash: the
same `doc_id::chunk_index` always maps to the same point id, so re-running ingest
*overwrites* every point instead of appending duplicates. Run `ingest.py` and it ingests
twice on purpose:

```
14 docs -> 56 chunks -> 56 points in 'ch06_handbook' (0.6s)
14 docs -> 56 chunks -> 56 points in 'ch06_handbook' (0.7s)
```

Fifty-six points after two runs, not 112. Without this, every deploy doubles your
collection and the same chunk appears four times in every top-k. (Location-based ids have
one weakness: if a document is edited so that it now has fewer chunks, the trailing old
chunks survive. Chapter 7 fixes that with content hashes and a manifest.)

**Create, then wrap** (Chapter 4's lesson): the collection is created with the embedding's
dimension probed at runtime, so switching `RAG_EMBED_MODEL` in `.env` does not break
anything as long as you `--recreate`.

## 6.3 Retrieve: k, filters, thresholds, context

```python
def retrieve(store, question, k=4, doc_id=None, score_threshold=None):
    flt = None
    if doc_id:
        flt = models.Filter(must=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id))])
    return store.similarity_search_with_score(question, k=k, filter=flt, score_threshold=score_threshold)
```

Three knobs a real retriever needs, and why they are parameters rather than constants:

- **`k`**: the caller knows the budget. A CLI can afford 6; an evaluation of recall@10
  needs 10; a tight API might use 3 after reranking (Chapter 9).
- **`doc_id` filter**: scope. "In the travel policy, what is…" or, in production, "only
  documents this tenant owns". Filters are exact and cheap; do them in the database, never
  by fetching 100 and discarding 95.
- **`score_threshold`**: the database-side "nothing relevant". Leave `None` until Chapter
  12 shows you how to calibrate it; a guessed threshold silently drops right answers.

`build_context()` is `format_docs` from `ragbook`: `[1] (source: file.md)\n<text>`, one
numbered block per chunk. The numbering is load-bearing: it is what citations point at.

```
Q: How many PTO days do I get?
[1] 0.547  02-pto-and-leave-policy.md  chunk 0
[2] 0.484  02-pto-and-leave-policy.md  chunk 1
[3] 0.439  14-faq.md  chunk 0
[4] 0.405  02-pto-and-leave-policy.md  chunk 2
```

`get_store()` attaches to the collection and **builds it if missing**. That makes every
downstream script runnable on a fresh machine (and in `QDRANT_MODE=memory`, where nothing
persists) without a manual step: a small thing that saves an hour a week.

## 6.4 Generate: the prompt grows a clause, the answer grows a receipt

Compared with Chapter 3, the prompt has one new sentence:

```
If two context items disagree, say so and prefer the one marked as a policy or more recent.
```

That is the fix for the silent conflict on q01. It is a *prompt* fix: cheap and partial.
The model now has permission and instruction to surface disagreement; whether it does so
reliably is measured in Chapter 11, and the structural fix (recency metadata, filtering
superseded documents at ingest) is Chapter 12.

`answer()` returns a dict, not a string:

```python
return {
    "question": question,
    "answer": response.text,
    "sources": [{"n": i, "source": ..., "chunk": ..., "score": ...} for ...],
    "usage": {"input_tokens": ..., "output_tokens": ...},
    "cost_usd": round(cost_usd(usage), 6),
}
```

Everything a caller could want to log, display, or evaluate is in there. The **cost** comes
from `response.usage_metadata` × a price table:

```python
PRICES = {"gpt-5.4-mini": (0.75, 4.50), ...}     # $ per 1M tokens, (input, output)
```

```
Q: What is the Beacon API rate limit?
A: The Beacon API rate limit is 600 requests per minute per API key [1][2][3].
   [1] 0.671  05-beacon-fleet-software.md (chunk 2)
   [2] 0.664  14-faq.md (chunk 2)
tokens {'input_tokens': 721, 'output_tokens': 24}  cost $0.000649
```

Sixty-five thousandths of a cent per question, 83% of it input tokens (they are 97% of
the *tokens*, but output tokens cost 6× more each). Ten thousand questions a day is $6.50. The retrieval side (one embedding call, ~15 tokens) is a rounding
error. When someone asks "what does RAG cost", this is the arithmetic.

## 6.5 The CLI, and the first three golden answers

```bash
uv run python code/ch06/rag_cli.py --golden 3
```

```
Q: How many days of PTO do full-time employees get per year?
A: Full-time employees receive **24 days of PTO per calendar year** [1].
   sources: [1] 02-pto-and-leave-policy.md, [2] 02-pto-and-leave-policy.md, [3] 14-faq.md, [4] 02-pto-and-leave-policy.md
   815 in / 19 out tokens, $0.000697
   reference: 24 days ... (the FAQ still says 20, but the 2026 policy supersedes it)

Q: How many unused PTO days can I carry over, and by when must I use them?
A: You can carry over up to **5 unused PTO days**, and you must use the carried-over days by **31 March** [1]

Q: How long is parental leave for a primary caregiver?
A: Primary caregivers get **26 weeks** of paid parental leave at full pay. ... [1]
total cost $0.00227
```

All three correct. And q01: *still* no mention of the FAQ's 20 days, despite the new
clause. Check the context before blaming anything: chunk 0 of the FAQ (747 characters)
*does* contain "20 days per year. (Updated February 2024.)" and it was retrieved at [3]. So
the model read both numbers and silently chose the policy's. It chose *correctly* (the FAQ
itself says "the policy document always wins") but it did not say so, and our clause did
not make it. One prompt sentence is a suggestion, not a guarantee; making conflict
reporting reliable needs either a structured step ("list any disagreements in the
context") or a grader that checks for it. Chapter 11 works this exact case.

## 6.6 What LangChain gave you, precisely

| You used | What it is | Why it earns its place |
|---|---|---|
| `Document` | text + metadata | one shape for every loader, splitter, store, retriever |
| `RecursiveCharacterTextSplitter` | chunking | well-tested, configurable separators, copies metadata |
| `QdrantVectorStore` | store adapter | `add_documents`/`similarity_search_with_score`/`as_retriever` are identical across 50+ databases |
| `ChatPromptTemplate` | prompt with variables | role-tagged messages, validated variables |
| `init_chat_model("openai:...")` | model constructor | change one string to switch provider |
| `prompt \| llm` (Runnable) | composition | `.invoke`, `.stream`, `.batch`, async, tracing (Ch 13) for free |
| `AIMessage.usage_metadata` | token accounting | standardised across providers |

What it did *not* give you, and never will: metadata conventions, id strategy, k,
thresholds, prompt wording, conflict policy, cost tables, evaluation. Those are the
system. Frameworks are 20% of a RAG product; the part interviewers probe is the other 80%.

## 6.7 [`ragbook/index.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/index.py): the same pipeline, packaged

From Chapter 7 on, every chapter starts with "an index of the handbook exists". Rather
than re-teach ingestion each time, [`ragbook/index.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/index.py) holds exactly the pipeline above:

```python
from ragbook import build_handbook_index
store = build_handbook_index("ch09_demo", chunk_size=800, chunk_overlap=120)   # builds if missing, else reuses
```

Read it once (under 90 lines): `chunk_documents` → create collection → `QdrantVectorStore` →
`add_documents(ids=chunk_id)`. It is `ingest.py` with the `load_sources` extension removed.

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch06/ingest.py                      # twice on purpose, same count
QDRANT_MODE=memory uv run python code/ch06/retrieve.py "How many PTO days do I get?"
QDRANT_MODE=memory uv run python code/ch06/generate.py "What is the Beacon API rate limit?"
QDRANT_MODE=memory uv run python code/ch06/rag_cli.py --golden 3
uv run python code/ch06/rag_cli.py --interactive                          # on-disk mode: ingests once, then persists
```

Expected output is quoted through §6.2–6.5. If you have already run Chapter 7's
`synthesize_docs.py`, `load_sources()` picks up `data/generated/` too and the first line
reads `19 docs -> 97 chunks -> 97 points` instead. Without `QDRANT_MODE=memory` the first
script that runs builds `.qdrant_data/collection/ch06_handbook/` and every later one reuses
it: note the missing "14 docs -> 56 chunks" line on the second run. (In memory mode each
`ingest()` call gets a fresh empty client, so the "twice on purpose" run only proves
idempotency in the on-disk or server modes.)

## Exercises

1. Run `retrieve.py` with `doc_id="14-faq"` for q01. Which chunk holds "20 days"? What
   chunk size would put it in the top 4 without the filter?
2. Add `--k` to `generate.py`'s CLI and plot cost vs k for k in {2, 4, 8, 12} on the
   `--golden 10` run. Where does the answer quality stop improving?
3. Add a `since` metadata field (the *Effective …* date each handbook file states) at load
   time and pass it through to `format_docs` so the model sees dates. Does q01 now mention
   the conflict?
4. Point `load_sources()` at a PDF of your own with `PyPDFLoader`. What metadata does each
   page carry? What did you have to add?
5. Replace `openai:` in `get_llm` with another provider you have a key for. What else had
   to change? (Answer: the price table.)

## Interview questions

**Q: How do you make ingestion idempotent?**
Derive point ids deterministically from content identity: `uuid5(doc_id::chunk_index)`
or a content hash: so re-ingesting upserts the same ids instead of appending. Keep a
manifest of what is indexed to delete chunks whose source disappeared. Never let a deploy
double the collection.

**Q: How do you handle document updates and deletions?**
Re-chunk the changed document; upsert chunks by stable id; delete any ids from that
document that are no longer produced (filter delete by `doc_id`, or diff against a
manifest). For deletions, filter-delete by `doc_id`. Schedule or trigger from the source
system's change events.

**Q: What metadata should every chunk carry?**
Source (path/URL), a stable document id, title or heading path, chunk index, and whatever
you will filter or cite on: dates (effective/updated), owner, classification/access
group, tenant, language, document type. Decide these at the loader boundary and enforce
them everywhere.

**Q: What does a RAG query cost, and where does the cost go?**
Roughly one embedding call (negligible) plus one chat completion whose input is dominated
by the retrieved context: k chunks × chunk tokens. With four 800-character chunks on
gpt-5.4-mini that is ~700 input tokens ≈ $0.0006 per question. k and chunk size are the
dials; output tokens are small unless you ask for long answers.

**Q: Where does the framework stop and your system begin?**
LangChain supplies containers (Document), adapters (loaders, stores, models), prompt
templates and composition. Your system is the metadata conventions, id and update
strategy, k/threshold/filter policy, prompt contract including conflict and refusal
behaviour, cost controls, and the evaluation harness. Interviews probe the latter.

**Q: Why return a dict from `answer()` instead of a string?**
Because the answer alone is unobservable. Sources let the UI show citations and let evals
compute retrieval metrics; usage lets you bill and alert; the question lets you log the
pair. Structured return values are how a prototype becomes a service.

**Q: A user says "the bot gave me the old PTO number". Walk me through the diagnosis.**
Print the retrieved chunks for that question. If the current policy chunk was not
retrieved: retrieval problem: chunking granularity, k, or the FAQ outranking it; fix with
metadata (effective date), filtering superseded docs, or reranking. If both were retrieved
and the model chose the old one: generation problem: strengthen the conflict clause,
present dates in the context, or grade sources before answering.

## Key takeaways

- Separate the write path (ingest) from the read path (retrieve, generate); they change,
  scale and fail independently.
- Stable ids make ingest idempotent; normalized metadata makes filters and citations
  possible; both are decided at the loader boundary.
- Expose k, filters and thresholds as parameters: the caller knows the budget.
- Return structured results with sources, usage and cost; the answer alone cannot be
  debugged or billed.
- The framework is the plumbing; the policies are the product.

## Next

→ [Chapter 7: More Data: Growing and Maintaining the Corpus](07-more-data.md)

Fifty-six chunks is not a corpus. Chapter 7 adds three novels of noise and five synthetic
policy documents, teaches incremental batched ingestion with a manifest, and shows what
more data does to retrieval: including what it does *not* do.
