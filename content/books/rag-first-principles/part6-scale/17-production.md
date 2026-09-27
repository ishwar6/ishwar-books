# Chapter 17 · Production: FastAPI, Async, Docker

> **Goal:** turn the pipeline into a service you could deploy tomorrow: a FastAPI app that
> loads its expensive clients once, answers asynchronously, streams tokens, caches, checks
> an API key and applies per-role retrieval filters; packaged in a Dockerfile and run next
> to a Qdrant server with Docker Compose. You will be able to answer the async/sync, DB
> session, auth and Dockerfile-vs-Compose questions cleanly.

New libraries in this chapter: **FastAPI** (web framework), **uvicorn** (the ASGI server),
**httpx** (async HTTP client), **Docker**. Everything RAG-related is unchanged from
Chapter 6.

---

## 17.1 The shape of the service: [`code/ch17/app.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch17/app.py)

```
client --POST /ask--> FastAPI --dependency: api key -> role
                         |--> Qdrant (async search, role filter)
                         |--> OpenAI (async generate, timeout)
                         '--> JSON {answer, sources, latency_ms, usage}
```

### Load once, in `lifespan`

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = build_handbook_index("ch17_handbook")     # Qdrant client + embeddings
    app.state.llm = get_llm(timeout=30, max_retries=2)
    yield
    app.state.store.client.close()
```

Building a Qdrant client, an embeddings client and an LLM client per request would add
hundreds of milliseconds and exhaust connections. `lifespan` runs once at startup and once
at shutdown. Anything expensive and shareable goes in `app.state`.

### Async endpoints: and what async actually means here

```python
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, role: str = Depends(current_role)):
    docs = await app.state.store.asimilarity_search(req.question, k=req.k, filter=role_filter(role))
    msg = await asyncio.wait_for(app.state.llm.ainvoke(prompt), timeout=40)
    ...
```

The interviewer asked *"when should you use async vs sync in FastAPI?"* and the candidate
drifted into message queues. The precise answer:

- A RAG request is **I/O-bound**: it waits ~50 ms on Qdrant and ~1–3 s on the LLM, doing
  almost no CPU work. While one request waits, the server should serve others.
- `async def` endpoint + `await` on **async clients** (`asimilarity_search`, `ainvoke`,
  `astream`) does that: the event loop parks the request at each `await` and runs another.
  One process handles many concurrent requests.
- `def` (sync) endpoint: FastAPI runs it in a **threadpool** (default 40 threads), so it
  also does not block the loop, but you are capped by threads and pay thread overhead. Fine
  for CPU-bound or legacy sync SDKs.
- The **mistake**: a *sync* SDK call (`llm.invoke`, `store.similarity_search`,
  `requests.get`) inside an `async def`. That blocks the whole event loop; every other
  request stalls for the duration. Either use the async method or wrap the sync call in
  `await run_in_threadpool(...)`.
- Message queues (the candidate's answer) solve a *different* problem: work that need not
  finish inside the HTTP request (ingestion, re-embedding, sending emails). Return 202 and
  process from a queue.

The client demo proves the point: five concurrent questions finished in ~2.4 s wall time
while their individual latencies summed to ~10 s.

### Streaming

```python
@app.post("/ask/stream")
async def ask_stream(...):
    async def gen():
        async for chunk in app.state.llm.astream(prompt):
            if chunk.text: yield chunk.text
    return StreamingResponse(gen(), media_type="text/plain")
```

Time-to-first-token is what users feel. `astream` yields tokens as they arrive; the
`StreamingResponse` forwards them. Retrieval still happens before the first token: keep it
fast (Chapter 18).

### Auth as a dependency, RBAC as a filter

```python
def current_role(x_api_key: str = Header(...)) -> str:
    role = API_KEYS.get(x_api_key)
    if not role: raise HTTPException(status_code=401, detail="invalid API key")
    return role
```

`Depends(current_role)` runs before the endpoint body; a bad key never reaches retrieval. In
production replace the dict with JWT verification: check signature, expiry and issuer, read
the role/tenant claims. The role becomes a Qdrant `Filter` (Chapter 16), so an employee
literally cannot retrieve the Confidential price book: the client demo shows the same
question answered as "I don't know" for `employee` and "$48,000" for `sales`.

**DB sessions** follow the same dependency pattern the interviewer asked about:

```python
def get_db():
    db = SessionLocal()
    try:
        yield db          # endpoint runs here
    finally:
        db.close()        # connection returns to the pool even on exceptions
```

A dependency with `yield` is per-request setup/teardown. For Qdrant we do not need it:
the client is a stateless pool held in `app.state`: but a SQLAlchemy session is per
request and must be closed.

### Caching

The app keys an in-memory dict on `sha256(role | k | normalised question)` with a TTL. That
catches exact repeats (the demo shows `cached=True latency=0 ms`). The next step is a
**semantic cache**: embed the question, look up a small Qdrant collection of past questions
with a high similarity threshold (≥0.95), return the stored answer. Both must include the
role/tenant in the key or you leak answers across users.

### Timeouts, limits, logs

`asyncio.wait_for(..., timeout=40)` → 504 instead of a hung connection. `get_llm(timeout=30,
max_retries=2)` handles transport retries. Rate limiting per key (token bucket in Redis, or
`slowapi`) protects your OpenAI quota. Logs are one JSON line per request with question,
role, latency and tokens: that is what you graph in Chapter 13's dashboards.

## 17.2 Dockerfile vs Docker Compose

The candidate struggled here. The one-sentence version:

> A **Dockerfile** is the recipe for building **one image** (how to install and start *your
> app*). **Docker Compose** is a YAML file that runs **several containers together** (your
> image plus off-the-shelf ones like Qdrant) with a shared network, volumes and
> environment, using one command.

[`code/ch17/Dockerfile`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch17/Dockerfile) (recipe for the API image):

```dockerfile
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project      # deps layer: cached until lock changes
COPY ragbook ./ragbook
COPY code ./code
COPY data ./data
RUN uv sync --frozen --no-dev
RUN useradd -m appuser && chown -R appuser /app
USER appuser                                            # never root in prod
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "code.ch17.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Two habits worth naming: copy the lock file and install dependencies *before* copying
source, so code changes do not invalidate the dependency layer; and run as a non-root user.

`docker-compose.yml` at the repo root (run the system):

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_storage:/qdrant/storage]
    healthcheck: { test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"], interval: 5s, retries: 10 }
  api:
    build: { context: ., dockerfile: code/ch17/Dockerfile }
    ports: ["8000:8000"]
    environment:
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      QDRANT_URL: http://qdrant:6333        # service name = DNS name on the compose network
    depends_on: { qdrant: { condition: service_healthy } }   # "started" is not "ready"
volumes:
  qdrant_storage:
```

Qdrant needs no Dockerfile: it is a stock image. The API finds it at `http://qdrant:6333`
because Compose gives every service a DNS name. The volume keeps vectors across restarts.
`depends_on` alone only waits for the Qdrant *container* to start, not for it to accept
connections: the health check plus `condition: service_healthy` closes that gap, otherwise the
API's lifespan can try to build the index before Qdrant is listening.
Secrets come from the environment, never baked into the image.

```bash
docker compose up -d          # build the api image, start both
docker compose logs -f api
docker compose down           # stop; add -v to also delete the volume
```

From here, Kubernetes is the same idea with more knobs (Deployments instead of services,
PersistentVolumeClaims instead of volumes, Secrets instead of env).

## Run it

Terminal 1:

```bash
QDRANT_MODE=memory uv run python code/ch17/app.py
```

Terminal 2:

```bash
uv run python code/ch17/client_demo.py
```

Observed:

```
health: {'status': 'ok', 'points': 56}

missing key -> 422
wrong key   -> 401

employee: I don't know based on the handbook.  (1283 ms, 745 tok)
   sales: $48,000 [3]  (1337 ms, 741 tok)

cached repeat: cached=True latency=0 ms

5 concurrent: wall=2448 ms, sum of individual latencies=10259 ms

streaming: Beacon's RTO is **30 minutes** and its RPO is **5 minutes**. [1]
[sources: 05-beacon-fleet-software.md, ...]
```

Server log, one JSON line per request:

```
{"t":"...","lvl":"INFO","msg":{"q": "hotel cap in Europe?", "role": "sales", "ms": 2444, "tokens": 720}}
```

To run in Docker (daemon required): `docker compose up -d`, then the same client demo.

## Exercises

1. Change `/ask` to a plain `def` and call the *sync* `similarity_search` / `invoke`. Re-run
   the 5-concurrent test. Then put the sync calls inside `async def` (the bug). Compare wall
   times and explain each.
2. Replace the API-key dict with JWT: `uv add pyjwt`, issue a token with a `role` claim,
   verify it in `current_role`.
3. Implement the semantic cache with a Qdrant collection `ch17_cache` (threshold 0.95).
   Measure hit rate on the golden set asked twice with paraphrases.
4. Add a `POST /ingest` that accepts a markdown file and upserts it: return 202 and do the
   work in a `BackgroundTask`. Why not do it inline?
5. Build the image and run Compose. Point `QDRANT_URL` at the container and confirm
   `ragbook.get_qdrant_client()` picks it up.

## Interview questions

**Q: Async vs sync in FastAPI: when do you use which?**
`async def` with awaited async clients for I/O-bound work (LLM, vector DB, HTTP) so one
process serves many concurrent requests. Plain `def` for CPU-bound work or sync-only SDKs;
FastAPI runs it in a threadpool. Never call a blocking sync SDK inside `async def`: it
freezes the event loop for everyone. Queues are for work that should not happen inside the
request at all.

**Q: How do you manage DB sessions in FastAPI?**
A dependency with `yield`: create the session, `yield` it to the endpoint, close it in
`finally` so the connection returns to the pool even on errors. Engine/pool is created once
at startup; sessions are per request.

**Q: How do you implement authentication?**
A dependency that reads the credential (API key header or `Authorization: Bearer <JWT>`),
validates it (lookup, or verify JWT signature/expiry/issuer), and returns the identity;
endpoints declare `Depends(...)`. Authorisation then becomes data: the role/tenant feeds
the retrieval filter.

**Q: Dockerfile vs Docker Compose?**
Dockerfile builds one image (install deps, copy code, define the start command). Compose
declares a multi-container application (your image plus stock images like Qdrant) with
networking (service names as DNS), volumes and environment, started with one command.

**Q: How do you reduce latency in a RAG API?**
Load clients once; async I/O; stream tokens; cache exact and semantic repeats; small `k`
with a reranker instead of large `k`; keep vectors in RAM (Chapter 18); a smaller model for
routing/grading; pre-compute embeddings for known FAQs.

**Q: What goes in logs for a RAG request?**
Request id, user/tenant, question (PII-scrubbed), retrieved chunk ids and scores, model,
token usage, latency per stage, cache hit, and any guard-rail decision. Enough to replay
the request and to compute cost per user.

## Key takeaways

- Build expensive clients once in `lifespan`; hold them in `app.state`.
- `async def` + async clients for I/O; sync SDK calls inside async endpoints are the classic
  performance bug.
- Auth is a dependency; authorisation is a retrieval filter.
- Cache with role in the key; stream for time-to-first-token; timeouts everywhere.
- Dockerfile = one image. Compose = the system (API + Qdrant) with DNS, volumes, env.

## Next

[Chapter 18: Terabyte scale](18-terabyte-scale.md): the arithmetic of a very large
corpus, quantization, HNSW tuning, multitenancy, sharding, and ingestion pipelines.
