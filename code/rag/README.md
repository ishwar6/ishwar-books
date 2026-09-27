# RAG: From First Principles (code)

The runnable code for the book at https://ishwar6.github.io/ishwar-books/books/rag-first-principles

```bash
cd code/rag
uv sync                      # Python 3.12 + every dependency, including the ragbook helpers
cp .env.example .env         # then set OPENAI_API_KEY
uv run python code/ch03/first_rag.py
```

| Path | What it is |
|---|---|
| `code/chNN/` | one runnable script per idea, for chapter NN ([index](code/README.md)) |
| `ragbook/` | shared helpers every chapter imports (`common.py`, `index.py`) |
| `data/handbook/` | the corpus: 14 documents from a fictional robotics company |
| `data/golden/` | the golden question set used for evaluation |
| `scripts/validate_all.py` | runs every script in memory mode and reports pass/fail |
| `docker-compose.yml` | Qdrant server and the Chapter 17 API |
