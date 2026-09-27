"""Chapter 25 - the two silent bake-off killers: missing prefixes and truncation.

Part 1 is the most expensive bug in this book. Several open models are trained
asymmetrically: the QUERY must carry an instruction prefix, the passage must not.
fastembed exposes `query_embed()`/`passage_embed()` which look like they handle
it. For these models they return vectors IDENTICAL to `embed()` - verified below
with np.allclose. Trust them and you will benchmark a good model at half its
quality, conclude it is bad, and ship the wrong choice.

Part 2 is truncation: every encoder has a maximum sequence length. Text past it
is silently dropped at INDEX time, so the answer is not in the vector at all and
no amount of query tuning finds it.

Run:  QDRANT_MODE=memory uv run python code/ch25/prefix_and_length.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "ch24"))
sys.path.insert(0, str(HERE.parents[1] / "ch10"))
from _embedders import QUERY_INSTRUCTION, Embedder, encode  # noqa: E402
from _shared import build_corpus, vague_queries  # noqa: E402
from metrics import ndcg_at_k, recall_at_k, unique_in_order  # noqa: E402

from ragbook import chunk_documents, load_golden, load_handbook  # noqa: E402

# (name, model_id, documented max input tokens, required query prefix)
MODELS = [
    ("arctic-embed-s", "snowflake/snowflake-arctic-embed-s", 512, QUERY_INSTRUCTION),
    ("bge-small-en-v1.5", "BAAI/bge-small-en-v1.5", 512, QUERY_INSTRUCTION),
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", 256, ""),
]


def score(dm: np.ndarray, qm: np.ndarray, doc_ids, golden) -> tuple[float, float]:
    dn = dm / np.clip(np.linalg.norm(dm, axis=1, keepdims=True), 1e-12, None)
    r5, nd = [], []
    for g, qv in zip(golden, qm):
        q = qv / max(float(np.linalg.norm(qv)), 1e-12)
        order = np.argsort(-(dn @ q))[:10]
        ids = unique_in_order([doc_ids[i] for i in order])
        rel = set(g["sources"])
        r5.append(recall_at_k(ids, rel, 5))
        nd.append(ndcg_at_k(ids, rel, 10))
    return float(np.mean(r5)), float(np.mean(nd))


def main() -> None:
    golden = [g for g in load_golden() if g["answerable"]]
    vague = vague_queries(golden)
    queries = [vague[g["id"]] for g in golden]
    chunks = build_corpus(hard=True)
    texts = [c.page_content for c in chunks]
    doc_ids = [c.metadata["doc_id"] for c in chunks]

    # --- 0. the root cause --------------------------------------------------
    from fastembed import TextEmbedding

    m = TextEmbedding(model_name="snowflake/snowflake-arctic-embed-s")
    probe = ["how much PTO do I get"]
    a = np.array(list(m.embed(probe)))
    b = np.array(list(m.query_embed(probe)))
    c = np.array(list(m.passage_embed(probe)))
    print("0. DOES THE LIBRARY APPLY THE PREFIX FOR YOU?")
    print(f"   embed() == query_embed()   : {np.allclose(a, b)}")
    print(f"   embed() == passage_embed() : {np.allclose(a, c)}")
    print("   Both True: the helpers are no-ops for this model in this version.")
    print("   Whatever your library promises, verify it with a comparison like this.\n")

    # --- 1. what the prefix is worth ---------------------------------------
    print("1. PREFIX ABLATION (realistic queries, 997-chunk corpus)")
    print(f"   {'model':<20}{'no prefix':>12}{'with prefix':>13}{'delta':>9}  card says")
    for name, model_id, _limit, prefix in MODELS:
        emb = Embedder(name, "fastembed", model_id, 384, query_prefix=prefix)
        dm = encode(emb, texts, "doc")
        q_no = encode(emb, queries, "query", with_prefix=False)
        q_yes = encode(emb, queries, "query", with_prefix=True)
        _, nd_no = score(dm, q_no, doc_ids, golden)
        _, nd_yes = score(dm, q_yes, doc_ids, golden)
        says = "REQUIRED" if prefix and name.startswith("arctic") else (
            "'not so necessary'" if prefix else "none (symmetric)")
        print(f"   {name:<20}{nd_no:>12.3f}{nd_yes:>13.3f}{nd_yes - nd_no:>+9.3f}  {says}")
    print(
        "\n   arctic-embed goes from the worst model in the bake-off to competitive with\n"
        "   a 3072-dim paid API model, on one string of boilerplate. The model card told\n"
        "   the truth both times: 'required' meant required, 'not so necessary' meant\n"
        "   worth 0.008. Read the card, then MEASURE the card.\n"
    )

    # --- 2. truncation at index time ---------------------------------------
    print("2. MAX SEQUENCE LENGTH - how much of your corpus is silently cut off")
    print("   First: never trust the documented limit. Encode something enormous and")
    print("   see where the tokenizer stops counting - that is the real window.\n")
    huge = "warehouse robot policy " * 2000
    clamps = {}
    for name, model_id, documented, _ in MODELS:
        enc = TextEmbedding(model_name=model_id)
        clamps[name] = enc.token_count([huge])
        flag = "" if clamps[name] == documented else "   <-- card and reality disagree"
        print(f"   {name:<20} documented {documented:>4} | measured {clamps[name]:>4}{flag}")

    print("\n   Note the trap in the numbers below: token_count() reports what the model")
    print("   WILL SEE, already truncated. A chunk sitting exactly at the clamp is a chunk")
    print("   that lost its tail. That is what makes truncation silent.\n")

    variants = {
        "800-char chunks (book default)": chunk_documents(load_handbook(), 800, 120),
        "2400-char chunks": chunk_documents(load_handbook(), 2400, 200),
    }
    for label, cs in variants.items():
        t = [c.page_content for c in cs]
        ids = [c.metadata["doc_id"] for c in cs]
        print(f"   {label}: {len(cs)} chunks")
        for name, model_id, _documented, prefix in MODELS:
            enc = TextEmbedding(model_name=model_id)
            counts = [enc.token_count([x]) for x in t]
            clipped = sum(1 for n in counts if n >= clamps[name])
            emb = Embedder(name, "fastembed", model_id, 384, query_prefix=prefix)
            dm = encode(emb, t, f"doc-{len(cs)}")
            qm = encode(emb, queries, "query")
            _, nd = score(dm, qm, ids, golden)
            print(f"      {name:<20} window {clamps[name]:>4} | median seen {int(np.median(counts)):>4} "
                  f"| truncated {clipped:>3}/{len(cs)} | nDCG@10 {nd:.3f}")
        print()
    print(
        "   Read the MiniLM row carefully. Its window is 128 tokens, so even the book's\n"
        "   default 800-character chunks are already being cut, and at 2400 characters it\n"
        "   indexes roughly the first fifth of every chunk. Its nDCG is not 'a weaker model'\n"
        "   so much as 'a model that never saw most of your corpus'.\n\n"
        "   Text past the window is not down-weighted; it is absent from the vector, at\n"
        "   INDEX time, forever. No query-side tuning recovers it. Chunk size and embedding\n"
        "   model are ONE decision, not two."
    )


if __name__ == "__main__":
    main()
