# Appendix C · Glossary

One or two lines per term. Chapter numbers point to where the term is used in earnest.

**Adaptive RAG**: Routing a question to no-retrieval, single-step or multi-step retrieval based on a predicted complexity. Ch 15.

**Agent**: An LLM in a loop that chooses actions (tool calls) based on observations until a stop condition. Ch 14–16.

**Agentic RAG**: RAG where retrieval is a tool the model decides to call, grade and repeat, rather than a fixed step. Ch 15.

**ANN (approximate nearest neighbour)**: Search that trades a little recall for speed by not comparing against every vector (HNSW, IVF). Ch 4.

**Answer correctness**: Agreement of the generated answer with a reference answer. App B.

**Answer relevance**: Whether the answer addresses the question asked, regardless of truth. App B.

**API key**: A secret token identifying a caller to a service; hashed at rest, rotatable. Ch 17.

**Atomic claim**: A single verifiable statement extracted from an answer for faithfulness checking. Ch 10, 11.

**Backoff (exponential)**: Retrying after waits that double each time, with jitter, to survive rate limits. Ch 16.

**Bi-encoder**: Two independent encoders (query, document) whose outputs are compared with cosine/dot; precomputable. Ch 2, 9.

**Binary quantisation**: Storing each vector dimension as one bit; ~32× smaller; needs rescoring. Ch 18.

**BM25**: Sparse lexical ranking function: IDF × saturated term frequency × length normalisation. Ch 8.

**Bootstrap confidence interval**: Resampling the per-question scores with replacement to get an interval around a metric; the honest way to report a small eval set. Ch 21.

**Bounding box (bbox)**: The rectangle giving a text block's or figure's position on a PDF page; the raw material for reconstructing reading order. Ch 28.

**Checkpointer**: LangGraph component that persists graph state per thread after every step (`InMemorySaver`, Postgres). Ch 14.

**Chunk**: A piece of a document small enough to embed and retrieve as a unit. Ch 5.

**Chunk overlap**: Characters/tokens shared between consecutive chunks so facts at boundaries are not split. Ch 5.

**Citation**: A reference from an answer sentence to the chunk that supports it, usually by numbered id. Ch 11.

**Cohen's kappa**: Agreement between two raters (your LLM judge and a human) corrected for chance agreement; how you calibrate a judge. Ch 26.

**ColBERT / late interaction**: Retrieval that keeps one vector per token and scores with MaxSim. Ch 19.

**Collection**: Qdrant's unit of storage: a set of points with the same vector configuration. Ch 4.

**ColPali**: Late-interaction retrieval over page images using a vision-language model; no OCR. Ch 19.

**CombSUM / CombMNZ**: Score-fusion baselines: sum normalised scores across retrievers; CombMNZ additionally multiplies by how many retrievers returned the document. Ch 23.


**Conditional edge**: A LangGraph edge whose target is chosen by a function of the state. Ch 14.

**Context engineering**: Deciding what goes into the model's context at each step (instructions, retrieved text, tool results, memory) under a token budget. Ch 19.

**Context precision / recall**: Judged usefulness of retrieved chunks / coverage of the reference answer by the context. App B.

**Context window**: Maximum tokens a model accepts in one call. Ch 1.

**Contextual retrieval**: Prepending an LLM-written situating sentence to each chunk before indexing (Anthropic, 2024). Ch 9.

**Cosine similarity**: Dot product of two vectors divided by their magnitudes; 1 = same direction. Ch 2.

**CRAG (Corrective RAG)**: Grade retrieval as correct/ambiguous/incorrect and take a corrective action (refine, web search). Ch 15.

**Cross-encoder**: A model that reads query and document together and outputs a relevance score; precise, slow; used for reranking. Ch 9.

**Dead-letter queue (DLQ)**: Where a work item goes after exhausting its retries, so one poison document cannot block a pipeline. Ch 30.

**Dense retrieval**: Retrieval by similarity of learned embeddings. Ch 2–4.

**Dimensions**: Length of an embedding vector (1,536 for text-embedding-3-small). Ch 2.

**Distance metric**: Cosine, dot product or Euclidean; chosen at collection creation. Ch 4.

**Document loader**: LangChain component that reads a source (PDF, directory, web) into `Document` objects. Ch 6.

**Dot product**: Σ aᵢbᵢ; equals cosine for unit vectors. Ch 2.

**ef_construction**: HNSW build-time beam width: how many candidates are considered when inserting a node. Higher means a better graph and slower indexing; fixed once built. Ch 22.

**ef_search (hnsw_ef)**: HNSW query-time beam width: how much of the graph a search explores. The one HNSW knob you can change without rebuilding; raises recall and latency together. Ch 22.

**Embedding**: A vector representation of text such that semantic similarity ≈ geometric closeness. Ch 2.

**Ensemble / hybrid retrieval**: Combining dense and sparse results, usually by RRF. Ch 8.

**Entailment (NLI)**: Deciding whether a claim is entailed, contradicted or simply not mentioned by a passage; a stricter basis for faithfulness than a yes/no 'supported' judgement. Ch 26.

**Eval (evaluation) suite**: Golden questions plus metrics run automatically after every change. Ch 10.

**Faithfulness / groundedness**: Fraction of answer claims supported by the retrieved context. Ch 10, 11.

**FastEmbed**: Qdrant's lightweight local embedding library; used here for sparse BM25 vectors. Ch 8.

**Few-shot prompting**: Including worked examples in the prompt to teach format and behaviour. Ch 20.

**Filter (payload filter)**: Qdrant condition on metadata applied during vector search (`must`, `should`, `must_not`). Ch 4.

**Fine-tuning**: Updating model weights on task data; good for style/format, not for injecting changing facts. Ch 1.

**Fusion**: Merging ranked lists from several retrievers (RRF, DBSF). Ch 8.

**Golden dataset**: Curated questions with reference answers and source ids used for evaluation. Ch 10.

**GraphRAG**: RAG over an LLM-extracted knowledge graph with community summaries for global questions. Ch 19.

**Grounding**: Constraining the model to answer only from provided context. Ch 11.

**Guardrail**: A check or limit around the model: PII filters, call limits, injection defences, human approval. Ch 16.

**Hallucination**: Output not supported by the context or the facts. Ch 11.

**Hard negative**: A training example that looks relevant but is not; sharpens embedding models. App A (DPR).

**Hit rate@k**: Whether any relevant chunk is in the top k. App B.

**HNSW**: Hierarchical Navigable Small World graph; the ANN index in Qdrant. Ch 4, 18.

**Human-in-the-loop**: Pausing an agent for approval before a consequential action (`interrupt`). Ch 16.

**HyDE**: Embedding a hypothetical LLM-written answer instead of the question. Ch 9.

**Idempotent upsert**: Writing with stable ids so re-running ingestion overwrites instead of duplicating. Ch 7.

**IDF (inverse document frequency)**: log(N/df); weights rare terms higher in BM25. Ch 8.

**In-memory vector store**: LangChain's `InMemoryVectorStore`; fine for a first RAG, not for production. Ch 3.

**Indirect prompt injection**: Instructions hidden in retrieved documents or tool outputs that the model may follow. Ch 19.

**Information bottleneck (bi-encoder)**: A bi-encoder must compress a document into one vector before seeing the query, so it encodes general topicality rather than the specific condition asked about. The reason rerankers exist. Ch 24.

**Ingestion**: The offline pipeline: load → parse → chunk → embed → store. Ch 6, 7.

**IVF**: Inverted file index: cluster vectors, search a few clusters. Ch 18.

**JWT**: Signed JSON token carrying identity and expiry, sent as a bearer token. Ch 17.

**k (top-k)**: Number of chunks retrieved (or returned after reranking). Ch 3.

**LangChain**: Framework providing models, messages, tools, documents, splitters, vector-store integrations and `create_agent`. Throughout.

**LangGraph**: Library for building stateful graphs (nodes, edges, checkpoints) that run agents. Ch 14–16.

**LangSmith**: Tracing and evaluation platform for LangChain/LangGraph runs. Ch 13.

**Late chunking**: Embedding the whole document first, then splitting the token embeddings into chunks. Ch 19.

**LazyGraphRAG**: GraphRAG variant that defers LLM summarisation to query time; indexing cost like vector RAG. Ch 19.

**Lease (work item)**: A time-limited claim on a queued item so a crashed worker's work is retried rather than lost. Ch 30.

**LLM-as-judge**: Using a model to grade outputs against a rubric; must be validated against humans. Ch 10.

**Long context**: Models accepting hundreds of thousands to millions of tokens; complement, not replacement, for RAG. Ch 19.

**Lost in the middle**: Models use information at the start/end of the context better than the middle. Ch 19.

**M (HNSW)**: Links kept per node in the HNSW graph. More links means better recall and more memory (about M x 2 x 4 bytes per vector) and slower construction. Ch 22.

**Matryoshka (MRL)**: Embeddings whose leading dimensions form a good smaller embedding; enables truncation. Ch 18, 19.

**MaxSim**: Late-interaction score: sum over query tokens of the max similarity to any document token. Ch 19.

**MCP (Model Context Protocol)**: Standard for exposing tools, resources and prompts to model-driven clients. Ch 19, 20.

**Metadata / payload**: Structured fields stored with a chunk (source, section, date, tenant) for filtering and citations. Ch 4.

**Middleware (LangChain agents)**: Hooks around model and tool calls: limits, retries, PII, summarisation. Ch 16.

**MinHash**: A sketch that estimates Jaccard similarity between documents in constant space; near-duplicate detection before you pay to embed. Ch 30.

**MMR (maximal marginal relevance)**: Retrieval that trades relevance against diversity to avoid near-duplicate chunks. Ch 9.

**MRR**: Mean of 1/rank of the first relevant result. App B.

**Multi-query**: Generating several rephrasings of the question, retrieving for each, and merging. Ch 9.

**Multi-tenancy**: Serving many customers from one system with strict data isolation via payload filters or shards. Ch 18.

**Multivector**: A Qdrant point holding many vectors (e.g. per token/patch) scored by MaxSim. Ch 19.

**nDCG**: Normalised discounted cumulative gain; ranking quality with position discount. App B.

**Node**: A LangGraph function that reads state and returns an update. Ch 14.

**nprobe**: The number of IVF clusters searched at query time; IVF's equivalent of ef_search, trading recall against latency. Ch 22.

**Observability**: Traces, logs and metrics that let you replay and diagnose a run. Ch 13.

**Overlap**: see Chunk overlap.

**Paired permutation test**: Significance test that shuffles which system produced which per-question score; correct for comparing two configs on the same eval set. Ch 21.

**Parent-document / small-to-big retrieval**: Retrieve small chunks for precision, return their parent section for context. Ch 9.

**Payload index**: Qdrant index on a metadata field to make filters fast (server mode). Ch 4, 18.

**PII**: Personally identifiable information; redact at ingest and on output. Ch 16, 17.

**Point**: A Qdrant record: id + vector(s) + payload. Ch 4.

**Precision@k**: Fraction of the top k that are relevant. App B.

**Product quantization (PQ)**: Splitting a vector into subvectors and replacing each with a codebook id, compressing 10-60x at some recall cost; often combined with a rescoring pass. Ch 22.

**Prompt caching**: Provider discount for repeated prompt prefixes; put stable content first. Ch 17, 19.

**Prompt injection**: Input crafted to make the model ignore its instructions; see Indirect prompt injection. Ch 19.

**Quantisation**: Storing vectors in fewer bits (int8, binary, product) to cut memory; rescore with originals. Ch 18.

**Query rewriting**: Transforming the user question before retrieval (rephrase, expand, decompose, step back). Ch 9.

**RAG**: Retrieval-Augmented Generation: fetch relevant text, put it in the prompt, generate a grounded answer. Ch 1.

**RAPTOR**: Tree of recursive cluster summaries retrieved at all levels. App A.

**Reading order**: The sequence a human would read a page in. Not stored in a PDF; reconstructed from block coordinates, and wrong reading order silently corrupts every downstream chunk. Ch 28.

**Recall@k**: Fraction of relevant chunks in the top k. App B.

**Recursion limit**: LangGraph cap on steps per run; stops runaway loops. Ch 16.

**Recursive character splitter**: Splits on a priority list of separators (headings, paragraphs, sentences) to a target size. Ch 5.

**Refusal**: The system saying it cannot answer from the available information. Ch 12.

**Reranker**: Second-stage model reordering the top candidates by relevance. Ch 9.

**Rescoring**: Recomputing similarity with full-precision vectors for the top quantised candidates. Ch 18.

**Retriever**: A component that returns documents for a query (`vector_store.as_retriever()` or a function). Ch 6.

**RRF (Reciprocal Rank Fusion)**: Fusion by Σ 1/(60 + rank). Ch 8.

**Self-RAG**: Model emits reflection tokens deciding retrieval and grading relevance/support. Ch 15.

**Semantic cache**: Cache keyed by embedding similarity of the question; returns stored answers for near-duplicates. Ch 19.

**Semantic chunking**: Splitting where embedding similarity between adjacent sentences drops. Ch 5.

**Shard**: A horizontal partition of a collection across nodes; custom shard keys for tenant isolation. Ch 18.

**SimHash**: A locality-sensitive fingerprint where similar documents get similar hashes, so near-duplicates can be found by Hamming distance. Ch 30.

**Sparse vector**: High-dimensional vector with few non-zeros (term → weight), e.g. BM25/SPLADE; stored natively in Qdrant. Ch 8.

**State (LangGraph)**: The typed dictionary passed between nodes; reducers (e.g. `add_messages`) define how updates merge. Ch 14.

**Step-back prompting**: Asking a more general question first to retrieve background, then the specific one. Ch 9.

**Structured output**: Forcing the model to return a schema (Pydantic) via `with_structured_output`. Ch 11, 15.

**Synthetic data**: LLM-generated questions/answers from your corpus, used to bootstrap evals. Ch 7.

**System prompt**: Instructions that frame the model's role and rules; stable prefix for caching. Ch 3.

**Token**: Sub-word unit of text; ~4 characters of English on average. Ch 1.

**Token bucket**: A rate limiter that refills tokens at a fixed rate; how you respect an embedding API's tokens-per-minute limit under concurrency. Ch 30.

**Tool**: A function the model can call, described by name, docstring and schema (`@tool`). Ch 14.

**ToolNode**: LangGraph prebuilt node that executes the tool calls in the last AI message. Ch 15.

**Trace**: Recorded tree of calls (model, retriever, tools) for one run, with inputs, outputs, latency and tokens. Ch 13.

**Usage metadata**: Token counts returned with a model response; basis for cost accounting. Ch 13.

**Vector database**: Store optimised for ANN search over embeddings with filters and payloads (Qdrant). Ch 4.

**Verbalised row**: A table row rewritten as a sentence ('The Atlas A2 carries up to 250 kg') so that natural-language queries match it. Ch 28.

**Weighted RRF**: Reciprocal rank fusion with a per-retriever weight, for when you know one retriever is more reliable on your corpus. Ch 23.

**Zero-shot**: Prompting with instructions only, no examples. Ch 20.
