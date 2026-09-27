"""ragbook - the tiny shared toolbox every chapter imports.

Read `ragbook/common.py` once (it is short). Everything else in this book is
plain, self-contained scripts under `code/chNN/`.
"""
from ragbook.common import (  # noqa: F401
    ROOT,
    DATA_DIR,
    HANDBOOK_DIR,
    GOLDEN_PATH,
    CHAT_MODEL,
    EMBED_MODEL,
    get_llm,
    get_embeddings,
    get_qdrant_client,
    load_handbook,
    load_golden,
    require_openai_key,
    format_docs,
    print_docs,
)
from ragbook.index import build_handbook_index, chunk_documents  # noqa: F401,E402
