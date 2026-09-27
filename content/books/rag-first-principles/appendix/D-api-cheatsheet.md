# Appendix D · API Cheatsheet (LangChain 1.x · LangGraph 1.x · langchain-qdrant · qdrant-client)

> Verified against the versions installed in this repo on 8 September 2026:
> `langchain 1.4.0`, `langchain-core 1.6.2`, `langchain-openai 1.6.0`, `langgraph 1.2.11`,
> `langchain-qdrant 1.1.0`, `qdrant-client 1.19.0`, `langchain-text-splitters 1.1.2`.
> Every import below was executed with `uv run python -c "..."` in this repo. If a tutorial
> you find online uses something from the **OLD → NEW** table at the end, it is out of date.

Keep this open while coding. Section D.1 is the 90% you use every day.

---

## D.1 The everyday imports

```python
# --- this repo's helpers (ragbook/common.py, ragbook/index.py)
from ragbook import (get_llm, get_embeddings, get_qdrant_client, load_handbook, load_golden,
                     format_docs, print_docs, build_handbook_index, chunk_documents)

# --- models
from langchain.chat_models import init_chat_model          # init_chat_model("openai:gpt-5.4-mini")
from langchain_openai import ChatOpenAI, OpenAIEmbeddings   # ChatOpenAI(model=...), OpenAIEmbeddings(model=..., dimensions=512)
from langchain.embeddings import init_embeddings           # init_embeddings("openai:text-embedding-3-small")

# --- messages, prompts, tools, agents
from langchain.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool, ToolRuntime, InjectedState
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import (ModelCallLimitMiddleware, ToolCallLimitMiddleware, PIIMiddleware,
    SummarizationMiddleware, ModelRetryMiddleware, ToolRetryMiddleware, HumanInTheLoopMiddleware,
    before_model, after_model, wrap_model_call, wrap_tool_call, dynamic_prompt, hook_config)

# --- documents, splitting, vector stores
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient, models
from langchain_community.retrievers import BM25Retriever     # (community pkg is sunset but works)
from rank_bm25 import BM25Okapi

# --- LangGraph
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Send, interrupt
from langgraph.store.memory import InMemoryStore

# --- observability
from langsmith import traceable, Client
from langchain_core.callbacks import UsageMetadataCallbackHandler
```

---

## D.2 Chat models

```python
llm = init_chat_model("openai:gpt-5.4-mini")          # provider prefix; kwargs go to ChatOpenAI
# NOTE: do not pass temperature= to GPT-5-family models; they reject it.

resp = llm.invoke("Reply OK")                          # str, or a list of messages, or [{"role":"user","content":...}]
resp.content            # str (or list of content blocks)
resp.text               # plain text convenience
resp.usage_metadata     # {'input_tokens':..,'output_tokens':..,'total_tokens':..}
resp.tool_calls         # [] or [{'name','args','id','type'}]

for chunk in llm.stream("..."):                        # streaming
    print(chunk.text, end="")

msgs = [SystemMessage("You are..."), HumanMessage("...")]
llm.invoke(msgs)

# structured output (Pydantic)
from pydantic import BaseModel, Field
class Grade(BaseModel):
    relevant: bool = Field(description="...")
llm.with_structured_output(Grade).invoke("...")       # -> Grade instance

# tools
@tool
def search(query: str) -> str:
    """One-line docstring: the model reads this to decide when to call it."""
    return "..."
llm_t = llm.bind_tools([search])                       # optional: tool_choice="search" to force it
ai = llm_t.invoke("...")
ai.tool_calls                                          # execute yourself, or let ToolNode / create_agent do it

# prompt | model
prompt = ChatPromptTemplate.from_messages([("system", "Use only <context>{context}</context>"), ("human", "{question}")])
chain = prompt | llm
chain.invoke({"context": "...", "question": "..."})
```

---

## D.3 Embeddings

```python
# `import ragbook` first (it loads .env); otherwise OpenAIEmbeddings raises "Missing credentials".
emb = OpenAIEmbeddings(model="text-embedding-3-small")             # 1536 dims; dimensions=512 to truncate (Matryoshka)
v = emb.embed_query("question text")                                # list[float]
vs = emb.embed_documents(["chunk 1", "chunk 2"])                    # list[list[float]]

from langchain_core.embeddings import DeterministicFakeEmbedding    # for tests / RAG_FAKE_EMBEDDINGS=1
sparse = FastEmbedSparse(model_name="Qdrant/bm25")                  # sparse BM25 vectors for hybrid
```

---

## D.4 Documents and splitting

```python
doc = Document(page_content="...", metadata={"source": "02-pto.md", "doc_id": "02-pto", "title": "PTO"})

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120,
                                          separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""])
chunks = splitter.split_documents(docs)              # keeps metadata
pieces = splitter.split_text(text)                   # list[str]

tok = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4o", chunk_size=300, chunk_overlap=30)
md  = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
md.split_text(markdown)                              # Documents with header metadata

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
PyPDFLoader("file.pdf").load()
```

---

## D.5 Qdrant: raw client

```python
client = QdrantClient(":memory:")                         # or QdrantClient(path="./.qdrant_data") or QdrantClient(url="http://localhost:6333", api_key=...)

client.create_collection(
    collection_name="c",
    vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE),
    # hybrid: vectors_config={"dense": VectorParams(...)}, sparse_vectors_config={"sparse": models.SparseVectorParams()}
    # quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True)),
    # hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),   # payload_m=16, m=0 for tenant-only indexes
)
client.collection_exists("c"); client.delete_collection("c"); client.count("c").count

client.upsert("c", points=[models.PointStruct(id="uuid-or-int", vector=[...], payload={"doc_id": "x"})])
client.create_payload_index("c", "doc_id", models.PayloadSchemaType.KEYWORD)      # server mode only
# tenant field:  field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True)

flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value="02-pto"))])
# match=models.MatchAny(any=[...]) ; range=models.Range(gte=..) ; must_not / should

res = client.query_points("c", query=[...], limit=5, query_filter=flt, with_payload=True,
                          search_params=models.SearchParams(hnsw_ef=128,  # exact=True for brute-force baseline
                              quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)))
res.points  # each: .id .score .payload
# NOTE: embedded/local Qdrant (":memory:" or path=) is brute-force: hnsw_ef / exact / quantization params only matter on a server.

# hybrid, server-side RRF
client.query_points("c",
    prefetch=[models.Prefetch(query=models.SparseVector(indices=[...], values=[...]), using="sparse", limit=20),
              models.Prefetch(query=[...dense...], using="dense", limit=20)],
    query=models.RrfQuery(rrf=models.Rrf(k=60)),           # or models.FusionQuery(fusion=models.Fusion.DBSF)
    limit=10)

client.delete("c", points_selector=models.FilterSelector(filter=flt))      # delete by filter
client.scroll("c", scroll_filter=flt, limit=100, with_payload=True)        # iterate
```

---

## D.6 Qdrant: LangChain wrapper

```python
store = QdrantVectorStore(client=client, collection_name="c", embedding=emb,
                          retrieval_mode=RetrievalMode.DENSE)          # collection must exist
# hybrid: embedding=emb, sparse_embedding=FastEmbedSparse(), retrieval_mode=RetrievalMode.HYBRID,
#         vector_name="dense", sparse_vector_name="sparse"
store = QdrantVectorStore.from_documents(docs, emb, location=":memory:", collection_name="c")   # creates collection
store = QdrantVectorStore.from_existing_collection(embedding=emb, collection_name="c", url="http://localhost:6333")

store.add_documents(chunks, ids=[...uuids...])                         # ids must be UUIDs or ints
store.similarity_search("q", k=5, filter=flt)                          # payload keys are "metadata.doc_id", "page_content"
store.similarity_search_with_score("q", k=5, score_threshold=0.3)
store.max_marginal_relevance_search("q", k=5, fetch_k=20, lambda_mult=0.5)
retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 5, "filter": flt})
#            search_type="mmr" | "similarity_score_threshold"
retriever.invoke("q")                                                   # -> list[Document]
store.delete(ids=[...]); store.client                                   # underlying QdrantClient
```

Payload layout written by the wrapper: `{"page_content": "...", "metadata": {...}}`: so
filters use `key="metadata.doc_id"`.

---

## D.7 BM25 and hybrid on the client side

```python
bm25 = BM25Retriever.from_documents(chunks, k=5)          # langchain_community; simple tokeniser
bm25.invoke("q")
BM25Okapi([doc.split() for doc in texts]).get_scores("q".split())     # raw rank_bm25

def rrf(*ranked_lists, k=60):
    scores = {}
    for lst in ranked_lists:
        for rank, d in enumerate(lst, start=1):
            scores[d.metadata["chunk_id"]] = scores.get(d.metadata["chunk_id"], 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)
# EnsembleRetriever / MultiQueryRetriever / ParentDocumentRetriever now live in langchain_classic.retrievers
```

---

## D.8 LangGraph

```python
from typing import Annotated, TypedDict, Literal

class State(TypedDict):
    messages: Annotated[list, add_messages]      # reducer: append instead of overwrite
    question: str
    docs: list

def retrieve(state: State) -> dict:              # node: takes state, returns a partial update
    return {"docs": store.similarity_search(state["question"], k=5)}

def route(state: State) -> Literal["generate", "rewrite"]:   # conditional edge fn returns next node name
    return "generate" if state["docs"] else "rewrite"

g = StateGraph(State)
g.add_node("retrieve", retrieve); g.add_node("generate", generate); g.add_node("rewrite", rewrite)
g.add_edge(START, "retrieve")
g.add_conditional_edges("retrieve", route)                         # or with mapping {"generate": "generate", "rewrite": "rewrite"}
g.add_edge("generate", END); g.add_edge("rewrite", "retrieve")
graph = g.compile(checkpointer=InMemorySaver())                    # checkpointer optional

config = {"configurable": {"thread_id": "t1"}, "recursion_limit": 25}
graph.invoke({"question": "...", "messages": []}, config)
for event in graph.stream(inputs, config, stream_mode="values"):   # "updates" | "messages" | "values"
    ...
graph.get_state(config)                                             # inspect checkpoint

# messages-only graphs
g = StateGraph(MessagesState)
g.add_node("agent", lambda s: {"messages": [llm.bind_tools(tools).invoke(s["messages"])]})
g.add_node("tools", ToolNode(tools))
g.add_conditional_edges("agent", tools_condition)                  # routes to "tools" or END
g.add_edge("tools", "agent")

# Command: update + goto in one; Send: fan-out
def node(state) -> Command[Literal["a", "b"]]:
    return Command(update={"x": 1}, goto="a")
def fan_out(state): return [Send("worker", {"item": i}) for i in state["items"]]

# long-term memory store
store = InMemoryStore()                                             # InMemoryStore(index={"embed": emb, "dims": 1536}) for semantic search
store.put(("user", "u1"), "pref", {"lang": "python"}); store.search(("user", "u1"), query="language")
```

---

## D.9 `create_agent` and middleware

```python
agent = create_agent(
    model=llm,                                   # or "openai:gpt-5.4-mini"
    tools=[search],
    system_prompt="Always call search before answering. Cite chunk ids.",
    middleware=[
        ModelCallLimitMiddleware(run_limit=6, exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="search", run_limit=3),
        ModelRetryMiddleware(max_retries=2),
        PIIMiddleware("email", strategy="redact"),
        SummarizationMiddleware(model="gpt-5.4-mini", trigger=("tokens", 4000), keep=("messages", 10)),
    ],
    response_format=Answer,                      # Pydantic -> result["structured_response"]
    checkpointer=InMemorySaver(),
)
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]},
                      config={"configurable": {"thread_id": "t1"}})
result["messages"][-1].text ; result.get("structured_response")

# custom middleware
@after_model
@hook_config(can_jump_to=["end"])
def require_retrieval(state: AgentState, runtime) -> dict | None:
    used_tool = any(getattr(m, "tool_calls", None) for m in state["messages"])
    if not used_tool: return {"messages": [AIMessage("I need to search first.")], "jump_to": "end"}

@wrap_model_call
def log_calls(request, handler):
    resp = handler(request); print(resp.result[0].usage_metadata if hasattr(resp, "result") else ""); return resp
```

---

## D.10 Observability

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-learning
```
```python
@traceable(name="rag_answer", tags=["v3"])
def answer(q): ...

cb = UsageMetadataCallbackHandler()
llm.invoke("...", config={"callbacks": [cb]}); cb.usage_metadata     # per-model token totals
```

---

## D.11 OLD → NEW (if you see the left column, the tutorial is stale)

| OLD (do not use) | NEW (Sep 2026) | Why |
|---|---|---|
| `from langchain.chains import RetrievalQA` / `RetrievalQA.from_chain_type(...)` | `prompt \| llm` with your own retrieval function, or `create_agent(tools=[retriever_tool])` | Chains package removed from `langchain` 1.x; explicit code is shorter and debuggable |
| `create_retrieval_chain`, `create_stuff_documents_chain`, `create_history_aware_retriever` | write the 5 lines: retrieve → `format_docs` → prompt → llm | same |
| `LLMChain(prompt=..., llm=...)`, `.run(...)`, `.predict(...)` | `(prompt \| llm).invoke({...})` | Runnable interface |
| `from langchain.agents import AgentExecutor, initialize_agent, create_react_agent, create_openai_tools_agent` | `from langchain.agents import create_agent` (LangGraph-based) | AgentExecutor is gone; `create_agent` returns a compiled graph |
| `from langgraph.prebuilt import create_react_agent` | `from langchain.agents import create_agent` | deprecated in favour of `create_agent` + middleware |
| `from langgraph.checkpoint.memory import MemorySaver` | `InMemorySaver` | renamed |
| `from langchain_community.vectorstores import Qdrant` / `Qdrant.from_documents` | `from langchain_qdrant import QdrantVectorStore` | old class is a thin legacy shim; no hybrid/sparse support |
| `client.search(collection, query_vector=...)` (qdrant-client) | `client.query_points(collection, query=..., using=...)` | `search` deprecated; `query_points` covers dense, sparse, hybrid, multivector |
| `client.recreate_collection(...)` | `if client.collection_exists(): client.delete_collection(); client.create_collection(...)` | deprecated |
| `from langchain_community.embeddings import OpenAIEmbeddings` / `langchain_community.chat_models.ChatOpenAI` | `from langchain_openai import OpenAIEmbeddings, ChatOpenAI` | provider packages |
| `from langchain_core.messages import ...` (still works) | `from langchain.messages import ...` | 1.x re-exports; either is fine, book uses `langchain.messages` |
| `from langchain.hub import pull("rlm/rag-prompt")` | write a `ChatPromptTemplate` | hub pulls hide the prompt you should own |
| `ChatOpenAI(model="gpt-4o-mini", temperature=0)` | `init_chat_model("openai:gpt-5.4-mini")`, no `temperature` | GPT-5 family rejects the parameter; gpt-4o-mini is legacy |
| `from langchain.text_splitter import ...` | `from langchain_text_splitters import ...` | separate package |
| `from langchain.document_loaders import ...` | `from langchain_community.document_loaders import ...` (or provider packages like `langchain-docling`) | moved |
| `from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever, ParentDocumentRetriever` | `from langchain_classic.retrievers import ...`: or write RRF / multi-query yourself (10 lines) | moved to the classic package |
| `from langchain_experimental.text_splitter import SemanticChunker` | implement semantic chunking with embeddings (Chapter 5) | experimental package not installed/maintained here |
| `ragas.evaluate(...)` (0.3/0.4) | metrics in [`code/ch10/metrics.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/metrics.py) | ragas currently breaks against langchain-community 0.4 |
| `agent.run("...")`, `chain("...")` | `.invoke(...)`, `.stream(...)`, `.ainvoke(...)` | uniform Runnable methods |
| `StateGraph(State, config_schema=...)` | `StateGraph(State, context_schema=...)` | renamed in 1.x |
| `graph.stream(..., stream_mode="debug")` for tokens | `stream_mode="messages"` | token streaming mode |
| `vectorstore.similarity_search(..., filter={"doc_id": "x"})` (dict filters) | `filter=models.Filter(must=[models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value="x"))])` | langchain-qdrant takes Qdrant filter objects |
| `OpenAI()` SDK calls inside LangChain code | `init_chat_model` / `OpenAIEmbeddings` | one abstraction, tracing and callbacks work |

---

## D.12 Ten-line sanity check

```bash
uv run python - <<'EOF'
from ragbook import get_llm, get_embeddings, build_handbook_index
import os; os.environ.setdefault("QDRANT_MODE", "memory")
store = build_handbook_index("sanity")
docs = store.similarity_search("How many PTO days?", k=2)
print([d.metadata["source"] for d in docs])
print(get_llm().invoke("Say OK").text)
EOF
```
If that prints two sources and `OK`, every import in this cheatsheet works on your machine.
