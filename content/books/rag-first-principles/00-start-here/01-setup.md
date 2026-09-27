# Setup

Five minutes. You need Python 3.12+ via `uv`, and an OpenAI API key.

## 1. Install dependencies

```bash
cd code/rag        # from a clone of github.com/ishwar6/ishwar-books
uv sync            # creates .venv on Python 3.12 and installs everything, including `ragbook`
```

If you do not have `uv` yet:
`curl -LsSf https://astral.sh/uv/install.sh | sh`.

Why 3.12 and not the system 3.14: several of these libraries do not publish 3.14 wheels yet.
`pyproject.toml` pins `requires-python = ">=3.12"` and `uv` picks 3.12 automatically.

## 2. Keys

```bash
cp .env.example .env     # if .env does not exist yet
```

Open `.env` and set `OPENAI_API_KEY`. Everything else has a working default. `.env` is
git-ignored; never commit it. [`ragbook/common.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/ragbook/common.py) loads it with `python-dotenv`.

Defaults and prices (per 1M tokens, official pricing page, Sep 2026):

| Setting | Default | Alternatives |
|---|---|---|
| `RAG_CHAT_MODEL` | `gpt-5.4-mini` ($0.75 in / $4.50 out) | `gpt-5-mini` ($0.25/$2.00), `gpt-5.6-luna` ($0.20/$1.20), `gpt-5.6-terra` ($2/$12) |
| `RAG_EMBED_MODEL` | `text-embedding-3-small` ($0.02, 1536-d) | `text-embedding-3-large` ($0.13, 3072-d) |

Do not pass `temperature` to the GPT-5 family: it is rejected. None of the book's code does.

## 3. Run one thing

```bash
uv run python code/ch03/first_rag.py
```

You should see three questions answered with cited sources. If you see
`OPENAI_API_KEY is not set`, step 2 did not take.

## 4. Qdrant: three modes, same code

`ragbook.get_qdrant_client()` picks the mode from `.env`:

| Mode | How | Use it for |
|---|---|---|
| **Embedded on disk** (default) | nothing to set; data in `./.qdrant_data` | reading the book. **One process at a time**: the embedded engine locks the directory. |
| **In memory** | `QDRANT_MODE=memory` | tests, throwaway experiments; re-embeds every run |
| **Server** | `docker compose up -d` then `QDRANT_URL=http://localhost:6333` | Chapter 17+, anything concurrent, payload indexes, quantization, the dashboard at http://localhost:6333/dashboard |

If two scripts run at once in embedded mode you will get a "storage folder is already accessed
by another instance" error. Either run them one at a time, or start the server. The same rule
applies *inside* one script: create one `QdrantClient` and pass it around: a second
`get_qdrant_client()` call in the same process fails the same way. The book's scripts and
`build_handbook_index(..., client=client)` follow this.

Docker Desktop must be running for `docker compose up`. Qdrant itself is a single container
(`qdrant/qdrant`) with a volume; see `docker-compose.yml`.

## 5. Optional: LangSmith tracing (Chapter 13)

Create a free account at https://smith.langchain.com, then in `.env`:

```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-learning
```

Every LangChain and LangGraph call is then traced automatically. All code works without it.

## 6. Running things

- Always from the repo root: `uv run python code/chNN/script.py`.
- [`code/ch10/run_eval.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch10/run_eval.py) accepts `--limit N` to run on a subset of the golden set while you iterate ([`code/ch13/cost_and_latency.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/rag/code/ch13/cost_and_latency.py) uses `--n N`).
- `uv run python scripts/validate_all.py` runs every script in the book in memory mode (costs about $1–2).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: ragbook` | run with `uv run python ...`, not bare `python`; or `uv sync` again |
| `storage folder ... already accessed` | another script is using embedded Qdrant; run one at a time or use the server |
| `Payload indexes have no effect in the local Qdrant` | harmless; the embedded mode ignores them (Chapter 4) |
| First run of a Chapter 8/9 script is slow | FastEmbed downloads a small sparse/rerank model once (~100 MB) |
| `RateLimitError` | the OpenAI client retries automatically; lower batch sizes in Chapter 7 if it persists |
| `libc++abi: terminating ... recursive_mutex lock failed` printed *after* a script's last line | a rare macOS shutdown race in a native library (onnxruntime/grpc, loaded by qdrant-client). The script's work and output are complete; the process crashed while exiting. Re-run if you need a clean exit code. Seen ~3 times in ~200 runs, mostly when several scripts run at once. |
| Fake results, want zero cost | `RAG_FAKE_EMBEDDINGS=1` gives deterministic random vectors: pipelines run, quality is meaningless |
