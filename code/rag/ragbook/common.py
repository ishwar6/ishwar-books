"""Shared helpers. Every chapter's code imports from here so the chapter
scripts stay short and focused on the ONE idea they teach.

What lives here (and nothing else):
  - paths to the sample data
  - `.env` loading + model names
  - get_llm()            -> a LangChain chat model  (OpenAI via init_chat_model)
  - get_embeddings()     -> a LangChain embeddings model (OpenAI)
  - get_qdrant_client()  -> a Qdrant client (embedded on disk by default, or a server)
  - load_handbook()      -> the sample corpus as LangChain Documents
  - load_golden()        -> the golden question/answer set used for evals
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- paths -----
ROOT = Path(__file__).resolve().parents[1]          # .../code/rag
DATA_DIR = ROOT / "data"
HANDBOOK_DIR = DATA_DIR / "handbook"                # 14 markdown docs, ~30 KB
GOLDEN_PATH = DATA_DIR / "golden" / "qa.yaml"       # ~40 Q/A pairs with sources

load_dotenv(ROOT / ".env")                          # never commit .env

# --------------------------------------------------------------- models -----
# Defaults are cheap. Override in .env (see .env.example for the price table).
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gpt-5.4-mini")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")


def require_openai_key() -> None:
    """Fail fast with a friendly message instead of a stack trace 40 lines deep."""
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set.\n"
            "  1. cp .env.example .env   (already done if .env exists)\n"
            "  2. paste your key into .env\n"
            "  3. re-run this script"
        )


def get_llm(model: str | None = None, **kwargs):
    """A chat model. `init_chat_model` is the LangChain 1.x way to build one;
    the "openai:" prefix picks the provider. kwargs pass straight to ChatOpenAI."""
    require_openai_key()
    from langchain.chat_models import init_chat_model

    return init_chat_model(f"openai:{model or CHAT_MODEL}", **kwargs)


def get_embeddings(model: str | None = None):
    """An embeddings model.

    Set RAG_FAKE_EMBEDDINGS=1 to get deterministic fake vectors (no API key,
    no cost). Retrieval quality is then random - use it only to check that a
    pipeline *runs*, never to judge results.
    """
    if os.getenv("RAG_FAKE_EMBEDDINGS") == "1":
        from langchain_core.embeddings import DeterministicFakeEmbedding

        return DeterministicFakeEmbedding(size=1536)

    require_openai_key()
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model=model or EMBED_MODEL)


def get_qdrant_client():
    """Qdrant, three ways - the same client API for all of them:

      QDRANT_URL set        -> a real server (Docker: `docker compose up -d`, or Qdrant Cloud)
      QDRANT_MODE=memory    -> in-process, gone when the script exits (tests)
      default               -> embedded, persisted to ./.qdrant_data (one process at a time!)
    """
    import atexit

    from qdrant_client import QdrantClient

    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None)
    if os.getenv("QDRANT_MODE", "local") == "memory":
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(path=str(ROOT / ".qdrant_data"))
    # Close the embedded engine *before* the interpreter tears down. Otherwise its
    # finaliser runs during shutdown and prints a harmless but ugly
    # "ImportError: sys.meta_path is None" traceback.
    atexit.register(client.close)
    return client


# ----------------------------------------------------------------- data -----
def load_handbook():
    """The sample corpus: every markdown file in data/handbook as a Document.

    metadata carries `source` (file name), `title` (first heading) and
    `doc_id` (file stem) - later chapters filter on these.
    """
    from langchain_core.documents import Document

    docs = []
    for path in sorted(HANDBOOK_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next((l.lstrip("# ").strip() for l in text.splitlines() if l.startswith("# ")), path.stem)
        docs.append(
            Document(
                page_content=text,
                metadata={"source": path.name, "doc_id": path.stem, "title": title},
            )
        )
    return docs


def load_golden() -> list[dict]:
    """Golden Q/A pairs. Each item: id, question, answer, sources (list of
    doc_ids that contain the answer), answerable (bool), keywords (list)."""
    import yaml

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------- printing -----
def format_docs(docs) -> str:
    """Turn retrieved Documents into one context string with numbered sources.
    The numbering lets the model cite [1], [2] ... and lets you check them."""
    return "\n\n".join(
        f"[{i}] (source: {d.metadata.get('source', '?')})\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )


def print_docs(docs, max_chars: int = 200) -> None:
    for i, d in enumerate(docs, start=1):
        snippet = d.page_content.replace("\n", " ")[:max_chars]
        print(f"  [{i}] {d.metadata.get('source', '?')}: {snippet}...")
