# Chapter 5 · Chunking, Measured

> **Goal:** you know the main chunking strategies (fixed, recursive, token-based,
> structure-aware, semantic), can implement semantic chunking in forty lines, and (most
> importantly) you stop arguing about chunk size and start *measuring* it against a
> golden set. You leave with a comparison table you produced yourself.

Code: [`code/ch05/chunking_lab.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch05/chunking_lab.py), [`code/ch05/semantic_chunker.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch05/semantic_chunker.py).

---

## 5.1 Why chunking is the highest-leverage decision

A chunk is the unit of three things at once:

1. **Embedding.** One vector per chunk. A chunk about two topics gets a vector that points
   between them and matches neither well.
2. **Retrieval.** The retriever returns chunks. If the answer is split across two chunks,
   you need both in the top-k: twice the luck required.
3. **Context.** Chunks fill the prompt. Big chunks bring the answer *and* a lot of
   unrelated text; the model has more to read and the bill is higher.

Too small: the chunk loses the context needed to understand it ("It costs $4,200": what
does?), tables split from their headers, sentences cut mid-thought. Too big: the vector is
blurry, the top-k is padded with irrelevant paragraphs, and you hit "lost in the middle".
There is no universal right size; there is a right size *for your documents and your
questions*, and you find it by measuring.

## 5.2 The strategies

**Fixed size**: `CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=0)`.
Split on one separator, pack pieces up to the size. Dumb and predictable. Its failure is
cutting wherever the count runs out.

**Recursive**: `RecursiveCharacterTextSplitter(chunk_size, chunk_overlap, separators=[...])`.
Try the first separator; any piece still too large is split by the next separator, and so
on. The default order is `["\n\n", "\n", " ", ""]` (paragraph, line, word, character)
so it *prefers* natural boundaries and only falls back to brutal cuts when it must. **The
order matters**: putting `" "` first would make every chunk a word soup. `ragbook.chunk_documents`
uses `["\n## ", "\n### ", "\n\n", "\n", " ", ""]`: it tries markdown headings first, which
is why the handbook chunks usually start at a section.

**Token-based**: `TokenTextSplitter(encoding_name="cl100k_base", chunk_size=200, chunk_overlap=30)`
or `RecursiveCharacterTextSplitter.from_tiktoken_encoder(...)`. Counts what the model
counts. Use this when you need a guaranteed token budget (embedding model limits, prompt
budgets). Character counts are a proxy: ~4 chars/token in English, far fewer in code or
CJK text.

**Structure-aware**: `MarkdownHeaderTextSplitter(headers_to_split_on=[("#","h1"),("##","h2")])`.
Splits at headings and **writes the heading path into metadata** (`h1: "Pricing and
Plans"`, `h2: "Discounts"`). That metadata can be embedded with the chunk or shown to the
model, restoring the context small chunks lose. There are equivalents for HTML, code and
JSON. Sections can still be too long, so cap them with a recursive splitter afterwards (the
lab does).

**Semantic**: split where the *meaning* changes. `semantic_chunker.py` implements it:

```python
sentences = split_sentences(doc.page_content)
vecs = normalize(embeddings.embed_documents(sentences))
sims = np.sum(vecs[:-1] * vecs[1:], axis=1)          # similarity of each sentence to the next
threshold = np.percentile(sims, 25)                   # the 25% biggest "drops" become cuts
# walk the sentences; start a new chunk where sim < threshold
```

Forty lines, and it *is* what the semantic-chunker classes in libraries do. Costs an
embedding call per sentence at ingest time. It shines on prose without headings
(transcripts, emails); on a handbook with headings it mostly rediscovers the headings,
worse.

**Parent/child ("small-to-big")**: embed small chunks for precise matching, but hand the
LLM the *parent* section they came from. Best of both; Chapter 9 implements it.

**Overlap**: repeating the tail of one chunk at the head of the next (`chunk_overlap`)
protects sentences and facts that straddle a boundary. 10–20% of chunk size is the usual
range. It costs storage and can return two near-identical chunks in the top-k (MMR helps).

## 5.3 The lab: measure, don't argue

`chunking_lab.py` runs each strategy over the handbook, indexes the chunks into a fresh
Qdrant collection, and answers the 42 answerable golden questions with plain top-5 search.
Three cheap metrics (Chapter 10 has the full set):

- **hit@5**: at least one of the top-5 chunks comes from a document that holds the answer.
- **hit@1**: the very first chunk does.
- **answer@3**: *all* the answer keywords for the question appear in the concatenated
  text of the top-3 chunks. This is the one that catches "the answer got split".

```
strategy           chunks  mean_tok   min   max   hit@5   hit@1  answer@3
fixed_500              73       108    40   158    1.00    0.95      0.88
recursive_800          55       148    13   225    1.00    0.98      0.95
recursive_300         162        50     2   102    1.00    0.98      0.86
tokens_200             51       178    41   200    1.00    0.98      0.98
markdown_headers      109        73    19   228    1.00    0.98      0.88
semantic              149        51     4   241    1.00    0.93      0.81
```

How to read this:

- **hit@5 is 1.00 everywhere.** With 14 documents, *some* chunk of the right document is
  always in the top 5. Document-level recall is too lenient to choose between strategies
  on a small corpus: a lesson about metrics, not chunking.
- **hit@1 separates the weak ones.** Fixed-size (0.95) and semantic (0.93) put the wrong
  document first more often. Fixed cuts land mid-table; semantic produces tiny fragments
  (min 4 tokens!) whose vectors are noise.
- **answer@3 is the metric that matters for generation**, and it spreads from 0.81 to
  0.98. `tokens_200` (mean 178 tokens, hard cap 200, 30 overlap) keeps the answer in one
  piece most often; `recursive_800` is close. `recursive_300` (0.86) loses nine points
  versus `recursive_800` (0.95): same algorithm, smaller pieces, more split answers.
- **Chunk count and mean size trade against each other.** 162 chunks of 50 tokens versus
  51 chunks of 178: three times the vectors, three times the storage, worse answers on this
  corpus. Smaller is not "more precise" by default.

The bottom of the run shows *why* fixed-size scored lower: the pricing table is cut
mid-row (`| Spa`), so the warranty and spare-battery rows land in the *next* chunk with no
header to say what the numbers mean.

Your corpus will produce a different table. **That is the point: run the lab on your
documents and your questions.** Ten minutes of measuring beats a month of opinions.

> "Why 800 and not 2,000?" is a designed experiment, not a preference.
> [Chapter 27 §27.5](../part8-deep-dives/27-retrieval-debugging.md) runs the full factorial sweep
(> size × overlap × strategy against recall, nDCG, faithfulness, tokens and cost) and gives the
> reasoning framework for where the optimum moves.

## 5.4 Rules of thumb (to be overridden by your measurements)

| Document type | Start with | Because |
|---|---|---|
| Handbooks, docs, wikis with headings | markdown/HTML header splitter → cap at 300–800 tokens | headings are the author's chunk boundaries |
| Long prose, transcripts | recursive 400–800 tokens, 10–15% overlap; try semantic | no structure to exploit |
| Tables, spec sheets | keep tables whole; one chunk per table with its caption; consider a row-per-chunk with the header prepended | a split table is unreadable |
| Code | language-aware splitter (function/class boundaries) | a half function embeds badly |
| FAQs, tickets, chat | one Q/A or one message per chunk | the natural unit |
| Slides, short pages | one page per chunk | already small |

And for the prompt side, the **context budget** math: with k = 5 chunks of 200 tokens the
context is ~1,000 tokens; at 800-token chunks it is 4,000. gpt-5.4-mini bills $0.75 per
million input tokens, so the difference is ~$0.002 per question: about $7/month at 100
questions a day, about $6,750/month at 100,000 a day. More important than money: models attend best to
the beginning and end of the context; a fact at token 3,000 of 4,000 is where "lost in the
middle" bites. Put the best chunk first (or first and last).

## 5.5 What to do with tables, images and links

Interviewers ask "a PDF with text, images and hyperlinks: how do you chunk it?" The
honest answer is that chunking is the *last* step and parsing is where the battle is:

- **Text**: extract with layout awareness (`pypdf` for simple PDFs; layout-aware parsers
  such as Docling or unstructured for multi-column and tables). Then structure-aware
  chunking.
- **Tables**: detect and keep whole; convert to markdown or CSV text so the header row
  stays attached; optionally also store a one-sentence LLM summary as the embedded text
  and the raw table as the payload the model receives.
- **Images**: run OCR for images of text; for diagrams and photos, have a vision model
  write a caption and embed the caption (or use a multimodal embedding model: Chapter 19).
  Store the image path in metadata so the answer can link to it.
- **Hyperlinks**: keep the URL in metadata and the anchor text in the chunk; if the linked
  page is part of your corpus, record the link as a relationship (graph RAG, Chapter 19).

## Run it

```bash
QDRANT_MODE=memory uv run python code/ch05/chunking_lab.py     # ~1 minute, six collections
uv run python code/ch05/semantic_chunker.py                    # just the PTO policy, semantic cuts
```

Expected: the table in §5.3 (your numbers may differ by a few hundredths: embeddings are
deterministic but ranking ties are not), then the truncated pricing table from
`fixed_500`. The semantic chunker prints its first six chunks of the PTO policy; look at
chunk 4 (57 characters: a table header separated from its rows) to see semantic
chunking's characteristic failure on structured text.

## Exercises

1. Add `recursive_1500` to the lab. Does answer@3 keep improving? What happens to
   `mean_tok`, and what would that do to prompt cost?
2. Change `chunk_overlap` in `recursive_300` from 50 to 150. Does answer@3 recover?
3. The markdown-header strategy has the heading path in `metadata["h2"]`. Prepend
   `"{h1} > {h2}\n"` to each chunk's `page_content` before indexing. Does hit@1 change?
4. Write a splitter that keeps markdown tables whole (detect lines starting with `|`) and
   otherwise behaves like recursive_800. Measure it.
5. Change the semantic chunker's percentile from 25 to 10 and 50. Plot chunks-count vs
   answer@3.

## Interview questions

**Q: What chunking strategy would you use for a PDF with text, images and hyperlinks?**
Parse first: layout-aware text extraction, tables kept whole and rendered as markdown with
their headers, OCR or vision captions for images, URLs preserved in metadata. Then
structure-aware recursive chunking (headings, then paragraphs) at 300–800 tokens with
10–15% overlap, and measure hit rate and answer coverage against a golden set before
settling on sizes.

**Q: How do you choose chunk size?**
Empirically. Build a small golden set, index the corpus under several sizes, measure
retrieval hit rate and whether the answer keywords survive in the top chunks, and check
prompt cost. Start around 300–800 tokens for prose; smaller for FAQs; whole units for
tables and code. Small chunks match precisely but lose context; large chunks blur the
embedding and dilute the prompt.

**Q: What does recursive character splitting actually do?**
It tries a list of separators in order (paragraph, newline, space, character) splitting
by the first, and recursively re-splitting any piece still over the size limit with the
next separator. It prefers natural boundaries and only cuts mid-word as a last resort.
The separator order is a design decision; putting headings first aligns chunks with
sections.

**Q: What is semantic chunking?**
Splitting where meaning shifts: embed sentences, compute similarity between consecutive
sentences, cut where the similarity drops below a threshold (often a percentile). Costs an
embedding per sentence at ingest; helps on unstructured prose; tends to over-fragment
structured documents that already have headings and tables.

**Q: Why chunk overlap?**
So a fact or sentence that straddles a boundary appears intact in at least one chunk.
Typically 10–20% of chunk size. Costs storage and can surface near-duplicate chunks, which
MMR or deduplication handles.

**Q: What is "lost in the middle"?**
The measured tendency of LLMs to use information at the start and end of a long context
better than information in the middle. Mitigations: fewer, better chunks (reranking), put
the top chunk first, and keep the total context far below the window size.

**Q: Your retrieval hit rate is 100% but answers are wrong. What do you look at?**
Whether the answer *text* is actually inside the retrieved chunks (coverage), not just
whether the right document was hit: the chunk may have cut the fact in half or separated
a table from its header. Then the prompt and the model. Document-level metrics hide
chunk-level failures.

## Key takeaways

- A chunk is simultaneously the unit of embedding, retrieval and context; its size trades
  precision against context and cost.
- Recursive splitting with headings first is the sane default; token-based when budgets
  matter; structure-aware when the document has structure; semantic for raw prose.
- Document-level hit@5 saturates on small corpora: measure hit@1 and answer coverage.
- On this corpus, ~200-token token-capped chunks and 800-char recursive chunks kept
  answers intact best; 300-char chunks and semantic fragments lost points.
- Parsing (tables, images, links) is where most "chunking" problems actually live.

## Next

→ [Chapter 6: A Generic, Reusable RAG Pipeline](06-generic-rag.md)

Chapter 6 turns everything so far into a small, reusable pipeline (ingest, retrieve,
generate, CLI) with stable ids, metadata filters, citations and cost accounting. It is
the code you would actually start a project from.
