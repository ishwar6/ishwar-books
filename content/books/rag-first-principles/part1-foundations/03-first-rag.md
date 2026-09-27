# Chapter 3 · Your First RAG (LangChain Only)

> **Goal:** you build, run and read every line of a complete retrieval-augmented QA system
> over the handbook: load → chunk → embed → store → retrieve → prompt → answer with
> citations. ~70 lines, LangChain + OpenAI only, nothing persisted. You will also watch
> it succeed, dodge an unanswerable question, and silently ignore a conflict: the last
> two are the subject of Part 4.

Code: [`code/ch03/first_rag.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch03/first_rag.py).

---

## 3.1 The shape of the program

```
build_store()                              answer(store, question)
  load_handbook()  -> 14 Documents           similarity_search_with_score -> 4 (chunk, score)
  split            -> 55 chunks              format_docs -> numbered context string
  embed + store    -> InMemoryVectorStore    PROMPT | llm -> AIMessage
```

Two functions, two pipelines. Keep them separate in your head; when the answer is wrong,
the first question is always *"which half broke?"*

## 3.2 Ingest, line by line

```python
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragbook import load_handbook, get_embeddings

def build_store() -> InMemoryVectorStore:
    docs = load_handbook()                                                   # 1
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)  # 2
    chunks = splitter.split_documents(docs)                                  # 3
    print(f"{len(docs)} documents -> {len(chunks)} chunks")
    return InMemoryVectorStore.from_documents(chunks, embedding=get_embeddings())  # 4
```

1. **`Document`** is LangChain's universal container: `page_content` (str) +
   `metadata` (dict). Every loader produces them; every splitter, store and retriever
   consumes them. `load_handbook()` makes one per markdown file with
   `metadata={"source", "doc_id", "title"}`.
2. **The splitter** cuts text into pieces of at most 800 *characters* (not tokens: this
   class counts characters unless you say otherwise), trying separators in order:
   paragraph break, newline, space, character. `chunk_overlap=120` repeats the tail of one
   chunk at the head of the next so a sentence cut in half still appears whole somewhere.
   Chapter 5 measures whether these numbers are good.
3. **`split_documents`** copies each document's metadata onto every chunk. That is how a
   chunk knows it came from `02-pto-and-leave-policy.md`: and how we cite it later.
4. **`InMemoryVectorStore.from_documents`** embeds every chunk (one batched API call) and
   keeps vector + text + metadata in a Python dict. It is the simplest vector store
   LangChain has; Chapter 4 swaps it for Qdrant with a two-line change and the rest of the
   program does not notice: that interchangeability is the point of the abstraction.

## 3.3 Query, line by line

```python
from langchain_core.prompts import ChatPromptTemplate
from ragbook import format_docs, get_llm

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions about the Lumora Robotics employee handbook.\n"
     "Use ONLY the numbered context below. If the context does not contain the answer, "
     "reply exactly: I don't know based on the handbook.\n"
     "Cite the context items you used like [1] or [2][3]. Be brief.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])

def answer(store, question, k=4):
    hits = store.similarity_search_with_score(question, k=k)      # [(Document, score)]
    chain = PROMPT | get_llm()                                     # Runnable pipe
    response = chain.invoke({"context": format_docs([d for d, _ in hits]), "question": question})
    return response.text
```

**The prompt is a contract**, and its three clauses each prevent a specific failure:

| Clause | Prevents |
|---|---|
| "Use ONLY the numbered context" | answering from training memory (which knows some *other* company's PTO policy) |
| "If the context does not contain the answer, reply exactly: I don't know…" | hallucination when retrieval came back empty-handed; the fixed phrase is also easy to detect in evals |
| "Cite … like [1]" | unverifiable answers; with numbered chunks you can check each claim against its source |

**`format_docs`** (in [`ragbook/common.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/common.py)) renders the chunks as
`[1] (source: file.md)\n<text>\n\n[2] …`. The numbering is what makes citations possible.

**`PROMPT | get_llm()`** is LangChain's Runnable composition: the output of the prompt (a
list of messages) becomes the input of the model. `chain.invoke(dict)` fills the template
variables and runs it. `get_llm()` is `init_chat_model("openai:gpt-5.4-mini")`: the
LangChain 1.x way to construct a chat model from a provider-prefixed name.

The response is an `AIMessage`. `.text` is the answer; `.usage_metadata` has the token
counts we print: get into the habit of looking at them.

## 3.4 What happened when we ran it

**q16: a clean hit.**

```
Q: What is the Beacon API rate limit?
   [1] 0.664  14-faq.md                    | **Q: What is the Beacon API rate limit?** A: 600 requests per minute p...
   [2] 0.607  05-beacon-fleet-software.md  | Services: `api`, `planner`, `fleet-gateway`, ...
A: 600 requests per minute per API key. [1][2]
   tokens: 758 in / 16 out
```

Two independent sources agree, both cited. 758 input tokens for four chunks: remember
that the context, not the question, is where your tokens go.

**u02: the unanswerable question, handled.**

```
Q: Who is the head of marketing at Lumora?
   [1] 0.619  01-company-overview.md
   [2] 0.566  02-pto-and-leave-policy.md
A: I don't know based on the handbook.
```

Note the scores: 0.619 for a chunk that does *not* contain the answer. Similarity search
**always returns k results**: it has no notion of "nothing relevant exists". The retriever
did its job (the company overview is the most *related* text); the prompt's "I don't know"
clause did the rest. Take that clause away and the model will happily invent a head of
marketing. Chapter 12 makes "not enough data" a first-class path instead of a prompt hint.

**q01: the conflict, silently resolved.**

```
Q: How many days of PTO do full-time employees get per year?
   [1] 0.584  02-pto-and-leave-policy.md   | # Paid Time Off (PTO) and Leave Policy *Effective 1 January 2026...
   [3] 0.490  14-faq.md                    | # Employee FAQ ... 20 days per year. (Updated February 2024.)
A: Full-time employees receive 24 days of PTO per calendar year. [1]
   reference: 24 days ... (the FAQ still says 20, but the 2026 policy supersedes it)
```

The answer is right: but look at what the model *did*: both the policy (24) and the
outdated FAQ (20) were in its context, and it picked one and said nothing. Today it picked
correctly, probably because the policy chunk was ranked first and reads as authoritative.
Change the chunking, the k, or the phrasing, and the FAQ might rank first. A system that
does not *surface* conflicts is a system that is sometimes confidently wrong. Chapter 6
adds a prompt clause for this; Chapter 11 measures it.

## 3.5 What LangChain gave you, and what it did not

Gave you: `Document`, a splitter, a vector store with an `embedding=` slot, a prompt
template with variables, `|` composition, and a model constructor that takes a provider
string. Swap `InMemoryVectorStore` for `QdrantVectorStore` or `openai:` for `anthropic:`
and nothing else changes.

Did not give you: any opinion about chunk size, k, the prompt wording, thresholds, or
whether the answer is right. Those are your job, and they are the rest of this book.

## Run it

```bash
uv run python code/ch03/first_rag.py
```

Expected (trimmed):

```
14 documents -> 55 chunks

Q: How many days of PTO do full-time employees get per year?
   [1] 0.584  02-pto-and-leave-policy.md  | # Paid Time Off (PTO) and Leave Policy ...
   [2] 0.515  02-pto-and-leave-policy.md  | ## Probation ...
   [3] 0.490  14-faq.md  | # Employee FAQ ...
   [4] 0.432  02-pto-and-leave-policy.md  | Parental leave can be taken in ...
A: Full-time employees receive 24 days of PTO per calendar year. [1]
   tokens: 856 in / 19 out

Q: What is the Beacon API rate limit?
A: 600 requests per minute per API key. [1][2]

Q: Who is the head of marketing at Lumora?
A: I don't know based on the handbook.
```

Three questions cost about a fifth of a cent.

## Exercises

1. Set `k=1`. Which of the three questions breaks first? Then `k=10`: what happens to
   the input token count and does the answer improve?
2. Delete the "I don't know" clause from the prompt and re-ask u02. Write down what the
   model invents.
3. Change `chunk_size=800` to `200`. Does the FAQ's "20 days" now outrank the policy's
   "24 days" for q01? Explain using the scores.
4. Add a fourth question of your own that requires two documents (e.g. golden q32, the
   120-robot price). Does k=4 bring in both?
5. Print `response.usage_metadata` for all three questions and compute the cost with
   gpt-5.4-mini's prices ($0.75 in / $4.50 out per 1M tokens). Where does the money go?

## Interview questions

**Q: Walk me through a minimal RAG implementation.**
Load documents into a common container with text plus metadata; split them into chunks of
a few hundred tokens with some overlap; embed each chunk and store vector, text and
metadata; at query time embed the question, take the top-k chunks by cosine similarity,
render them as numbered context in a prompt that instructs the model to answer only from
that context, say "I don't know" otherwise, and cite chunk numbers; call the model and
return the answer with the sources. Print the retrieved chunks and scores every time so you
can tell retrieval failures from generation failures.

**Q: Why does the retriever return results for a question the corpus cannot answer?**
Similarity search ranks by nearness and always returns k items; a score of 0.6 means "the
most related thing we have", not "this contains the answer". You need either a calibrated
score threshold, an explicit relevance grader, or a prompt that permits "I don't know":
ideally all three.

**Q: Where do the tokens go in a RAG call?**
Almost entirely into the retrieved context: four 800-character chunks are ~750 tokens
versus ~15 for the question and ~20 for the answer. k and chunk size are your cost dials.

**Q: What does the prompt need to contain in a RAG system?**
A scope statement, an instruction to use only the provided context, an explicit permitted
fallback ("I don't know based on …"), a citation format tied to numbered chunks, and a
brevity constraint. Optionally: how to handle contradictions and recency.

**Q: Your RAG answered correctly but you say it is still unsafe. Why?**
Because it resolved a contradiction between two retrieved chunks without saying so. The
same system with slightly different ranking would give the other answer with the same
confidence. Correctness on one run is not reliability; you need conflict handling and an
evaluation set.

**Q: What does `prompt | llm` do in LangChain?**
It composes two Runnables: the prompt template turns a dict of variables into a list of
chat messages, and the model turns messages into an `AIMessage`. `invoke` runs the pipe;
the same object also supports `stream`, `batch` and async variants for free.

## Key takeaways

- A complete RAG is ~70 lines: two pipelines, six steps, one prompt.
- The `Document` container and the vector-store abstraction let you swap storage and
  providers without touching the logic.
- The prompt is a contract with three clauses: only context, permitted "I don't know",
  citations.
- Retrieval always returns k results; "relevant" is your problem, not the database's.
- A correct answer that hid a contradiction is a latent bug. Print the chunks. Always.

## Next

→ [Chapter 4: Qdrant: A Real Vector Database](../part2-building/04-qdrant.md)

The in-memory store vanishes when the script exits and can only filter with a Python
callback that scans every document. Chapter 4 replaces it with Qdrant: first through the raw client so you see what a vector database
actually stores, then through LangChain's wrapper.
