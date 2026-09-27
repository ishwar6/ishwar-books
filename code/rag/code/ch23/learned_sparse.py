"""Chapter 23 - learned sparse retrieval: keeping the inverted index, learning the weights.

    QDRANT_MODE=memory uv run python code/ch23/learned_sparse.py

BM25 weights a term by corpus statistics (how rare it is, how often it repeats).
SPLADE weights it with a transformer, and - the important part - it can put weight
on terms that DO NOT APPEAR in the text at all. That is vocabulary mismatch solved
while staying a sparse vector, so it still goes in an inverted index and still
supports exact-term matching.

Downloads ~530 MB of model on first run (cached afterwards).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrievers import BM25, dense_scores, evaluate, load_corpus   # noqa: E402

SPLADE = "prithivida/Splade_PP_en_v1"
BM25_SPARSE = "Qdrant/bm25"


def get_model(name: str):
    try:
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(model_name=name)
    except Exception as e:                                  # offline reader
        print(f"could not load {name}: {type(e).__name__}: {str(e)[:120]}")
        print("(needs one network download; skipping the learned-sparse comparison)")
        return None


def sparse_vectors(model, texts: list[str]) -> list[dict[int, float]]:
    """{term_id: weight} per text - exactly what goes into an inverted index.
    (Dense-ifying is not an option: Qdrant's BM25 model hashes terms into a
    32-bit id space, so the 'vocabulary' has 4 billion slots.)"""
    return [{int(i): float(v) for i, v in zip(e.indices, e.values)}
            for e in model.embed(texts)]


def sparse_scores(qs: list[dict[int, float]], ds: list[dict[int, float]]) -> np.ndarray:
    """Sparse dot product by intersecting the smaller dict - the inverted-index
    operation, written out."""
    S = np.zeros((len(qs), len(ds)), dtype=np.float32)
    for i, q in enumerate(qs):
        for j, d in enumerate(ds):
            small, large = (q, d) if len(q) < len(d) else (d, q)
            S[i, j] = sum(w * large[t] for t, w in small.items() if t in large)
    return S


def show_expansion(model, tokenizer, query: str) -> None:
    emb = list(model.embed([query]))[0]
    order = np.argsort(-emb.values)
    terms = [(tokenizer.decode([int(emb.indices[j])]), float(emb.values[j])) for j in order]
    q_words = set(query.lower().replace("?", "").split())
    print(f'query: "{query}"')
    print(f"  SPLADE produces {len(emb.indices)} weighted terms:")
    present = [(t, w) for t, w in terms if t in q_words][:6]
    added = [(t, w) for t, w in terms if t not in q_words and not t.startswith("##")][:8]
    print("    from the query : " + ", ".join(f"{t}({w:.2f})" for t, w in present))
    print("    ADDED by model : " + ", ".join(f"{t}({w:.2f})" for t, w in added))


def main() -> None:
    chunks, texts, doc_ids, golden = load_corpus()
    questions = [g["question"] for g in golden]
    D = dense_scores(texts, questions)
    bm = BM25(texts)
    B = np.stack([bm.scores(q) for q in questions])

    splade = get_model(SPLADE)
    if splade is None:
        return

    print("=" * 74)
    print("1. term expansion: what a learned sparse model adds")
    print("=" * 74)
    tok = splade.model.tokenizer
    for q in ("How long does a full charge of an Atlas A2 robot take?",
              "What is the rate limit for the Beacon API?"):
        show_expansion(splade, tok, q)
        print()
    print("""The added terms are the point. A chunk saying "90 minutes to charge" shares
no rare term with a question asking about "runtime", but SPLADE puts weight on
'time', 'hour', 'last' in BOTH, so the inverted index matches them. BM25 cannot
do this: its vocabulary is exactly the tokens present.""")

    print("=" * 74)
    print("2. retrieval quality: BM25 vs learned sparse vs dense")
    print("=" * 74)
    print("encoding corpus with SPLADE...")
    S_docs = sparse_vectors(splade, texts)
    SP = sparse_scores(sparse_vectors(splade, questions), S_docs)

    bm25_sparse = get_model(BM25_SPARSE)
    rows = {"dense (OpenAI)": D, "BM25 (ours)": B, "SPLADE++": SP}
    nnz = {"dense (OpenAI)": 1536,
           "SPLADE++": int(np.mean([len(d) for d in S_docs]))}
    if bm25_sparse is not None:
        Db = sparse_vectors(bm25_sparse, texts)
        rows["BM25 (Qdrant model)"] = sparse_scores(sparse_vectors(bm25_sparse, questions), Db)
        nnz["BM25 (Qdrant model)"] = int(np.mean([len(d) for d in Db]))

    print(f"\n{'retriever':<22} {'recall@5':>9} {'MRR':>7} {'nDCG@5':>8} {'nnz/doc':>9}")
    print("-" * 74)
    for name, S in rows.items():
        m = evaluate(lambda qi, S=S: list(np.argsort(-S[qi])), doc_ids, golden, k=5)
        n = nnz.get(name)
        print(f"{name:<22} {m['recall@5']:>9.3f} {m['mrr']:>7.3f} {m['ndcg@5']:>8.3f} "
              f"{(str(n) if n else '-'):>9}")

    print("\n" + "=" * 74)
    print("3. the identifier test - does expansion cost exact matching?")
    print("=" * 74)
    probes = [("BEACON-4187", "11-release-notes-beacon-4.2"),
              ("beaconctl rollout undo", "08-runbook-fleet-outage"),
              ("ISO 3691-4", "04-atlas-a2-specification"),
              ("Veldmark", "12-postmortem-2026-03-rotterdam-halt")]
    Dp = dense_scores(texts, [p[0] for p in probes], tag="idprobe")
    Sp = sparse_scores(sparse_vectors(splade, [p[0] for p in probes]), S_docs)
    print(f"{'query':<24} {'dense r':>8} {'BM25 r':>7} {'SPLADE r':>9}")
    print("-" * 74)
    for j, (q, gold_doc) in enumerate(probes):
        bs = bm.scores(q)

        def rank_of(scores, gold_doc=gold_doc):
            for r, i in enumerate(np.argsort(-scores), start=1):
                if doc_ids[i] == gold_doc:
                    return r
            return 999

        print(f"{q:<24} {rank_of(Dp[j]):>8} {rank_of(bs):>7} {rank_of(Sp[j]):>9}")

    print("""
Note what actually happened on the identifier rows: SPLADE is WORSE than plain
BM25 there (BEACON-4187 at rank 8 vs rank 1). Two reasons, and both are worth
knowing before you adopt it:

  - its vocabulary is BERT wordpiece, so 'BEACON-4187' is shredded into subwords
    and the rare-token signal that BM25 relies on is gone;
  - expansion adds terms that are merely associated, not synonymous. For "Beacon
    API rate limit" above it added 'price', 'money', 'fee' and 'lighthouse' -
    it read 'rate' as a tariff and 'beacon' as a lighthouse. Those weights pull
    in genuinely irrelevant chunks.

So learned sparse is not a strict upgrade over BM25; it trades exact-match
precision for vocabulary reach. It also costs more: ~123 non-zero terms per doc
against BM25's ~48, so longer posting lists, plus a transformer forward pass per
document at ingest and per query at search time.

When to reach for it: vocabulary mismatch is your measured failure mode and you
would rather run one inverted index than a vector DB. When not to: you have
identifiers, codes or commands in your corpus (keep BM25), or you already run a
dense index plus a reranker - the reranker fixes most of what SPLADE would fix,
with less operational surface.""")

    print("=" * 74)
    print("4. storing sparse vectors in Qdrant (the API, same as Chapter 8)")
    print("=" * 74)
    print("""
    client.create_collection(
        collection_name="ch23_sparse",
        vectors_config={},                          # no dense vectors at all
        sparse_vectors_config={"text": models.SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False))},
    )
    client.upsert("ch23_sparse", points=[models.PointStruct(
        id=i, payload={"doc_id": d},
        vector={"text": models.SparseVector(indices=e.indices.tolist(),
                                            values=e.values.tolist())})
        for i, (e, d) in enumerate(zip(model.embed(texts), doc_ids))])

    client.query_points("ch23_sparse", using="text",
                        query=models.SparseVector(indices=qi, values=qv), limit=5)

  Swap the model name and nothing else changes: to Qdrant, BM25 and SPLADE are both
  just {index: weight} maps. The choice of who computes the weights is yours.""")


if __name__ == "__main__":
    main()
