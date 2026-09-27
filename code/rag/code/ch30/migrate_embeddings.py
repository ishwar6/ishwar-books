"""Chapter 30 - changing your embedding model without downtime.

    QDRANT_MODE=memory uv run python code/ch30/migrate_embeddings.py
    QDRANT_MODE=memory uv run python code/ch30/migrate_embeddings.py --tolerance 0.0

Sooner or later you change embedding model - a better one ships, or you shrink
dimensions to halve your RAM bill. You cannot mix: vectors from two models are
not comparable, so a half-migrated collection returns nonsense for exactly the
queries that touch the new half.

The pattern, in order, and none of the steps is optional:

    v1 (live)  <--- alias "handbook_live" <--- your application
    v2 (dark)  <--- backfill, every point marked embed_version=v2
       |
       +-- validate v2 on the golden set BEFORE anyone sees it
       +-- flip the alias (atomic: readers never see a half-built index)
       +-- keep v1 until you are sure; the rollback is another flip

Here v1 is text-embedding-3-small at 1536 dims and v2 is the same model at 512
dims (Matryoshka truncation, Chapter 2) - a 3x storage cut, and a real decision
you would want evidence for before shipping.
"""
from __future__ import annotations

import argparse
import uuid

import numpy as np
from qdrant_client import models

from ragbook import (chunk_documents, get_embeddings, get_qdrant_client,
                     load_golden, load_handbook)

ALIAS = "handbook_live"
NAMESPACE = uuid.UUID("6f1c2a3e-0000-4000-8000-0000000030b2")


def build(client, name: str, chunks, embedder, version: str, batch: int = 64) -> int:
    """Build one collection, marking every point with the version that made it.

    The marker is what makes a backfill resumable and auditable: you can always
    ask 'how many points are still on v1?' and restart where you stopped.
    """
    dim = len(embedder.embed_query("probe"))
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(
        size=dim, distance=models.Distance.COSINE))

    for start in range(0, len(chunks), batch):
        window = chunks[start:start + batch]
        vectors = embedder.embed_documents([c.page_content for c in window])
        client.upsert(name, points=[
            models.PointStruct(
                id=str(uuid.uuid5(NAMESPACE, f"{c.metadata['doc_id']}::{c.metadata['chunk_index']}")),
                vector=vec,
                payload={"page_content": c.page_content,
                         "metadata": c.metadata,
                         "embed_version": version})
            for c, vec in zip(window, vectors)])
    return dim


def evaluate(client, name: str, embedder, golden, chunks, k: int = 5) -> tuple[float, float]:
    """(hit@1, MRR) at document level - the gate the migration has to pass."""
    doc_ids = {}
    for c in chunks:
        doc_ids[str(uuid.uuid5(NAMESPACE, f"{c.metadata['doc_id']}::{c.metadata['chunk_index']}"))] = \
            c.metadata["doc_id"]

    hit1 = rr = 0.0
    vectors = embedder.embed_documents([g["question"] for g in golden])
    for item, vec in zip(golden, vectors):
        hits = client.query_points(name, query=vec, limit=k).points
        ranked = [h.payload["metadata"]["doc_id"] for h in hits]
        gold = set(item["sources"])
        if ranked and ranked[0] in gold:
            hit1 += 1
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in gold:
                rr += 1 / rank
                break
    return hit1 / len(golden), rr / len(golden)


def alias_targets(client) -> list[str]:
    return [a.collection_name for a in client.get_aliases().aliases if a.alias_name == ALIAS]


def point_alias(client, collection: str) -> None:
    """One atomic call: readers go from all-v1 to all-v2 between two queries."""
    client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=collection, alias_name=ALIAS))])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.02,
                    help="how much MRR may drop before the migration is refused")
    ap.add_argument("--new-dims", type=int, default=512)
    args = ap.parse_args()

    client = get_qdrant_client()
    chunks = chunk_documents(load_handbook())
    golden = [g for g in load_golden() if g["answerable"]]

    from langchain_openai import OpenAIEmbeddings

    from ragbook import EMBED_MODEL

    old = get_embeddings()                                     # 1536 dims
    new = OpenAIEmbeddings(model=EMBED_MODEL, dimensions=args.new_dims)

    print(f"corpus: {len(chunks)} chunks, gate: {len(golden)} golden questions\n")

    dim_v1 = build(client, "ch30_v1", chunks, old, "v1")
    point_alias(client, "ch30_v1")
    print(f"v1 built: {client.count('ch30_v1').count} points at {dim_v1} dims")
    print(f"alias {ALIAS} -> {alias_targets(client)}  (this is what production reads)")

    print(f"\nbackfilling v2 at {args.new_dims} dims, in the dark...")
    dim_v2 = build(client, "ch30_v2", chunks, new, "v2")
    print(f"v2 built: {client.count('ch30_v2').count} points at {dim_v2} dims "
          f"({dim_v1/dim_v2:.1f}x smaller vectors)")
    versions = {p.payload["embed_version"]
                for p in client.scroll("ch30_v2", limit=100)[0]}
    print(f"every point marked embed_version={versions} - "
          f"a resumable backfill asks for the ones that are not")

    print("\nvalidating BEFORE the flip:")
    h1_old, mrr_old = evaluate(client, "ch30_v1", old, golden, chunks)
    h1_new, mrr_new = evaluate(client, "ch30_v2", new, golden, chunks)
    print(f"{'collection':>12} {'dims':>6} {'hit@1':>7} {'MRR':>7}")
    print(f"{'v1 (live)':>12} {dim_v1:6} {h1_old:7.3f} {mrr_old:7.3f}")
    print(f"{'v2 (dark)':>12} {dim_v2:6} {h1_new:7.3f} {mrr_new:7.3f}")
    delta = mrr_new - mrr_old
    if mrr_old >= 0.999 and mrr_new >= 0.999:
        print("  (both saturated on this 14-document corpus: a gate that cannot move")
        print("   cannot protect you - size the golden set to the decision, Chapter 21)")
    print(f"{'delta':>12} {'':6} {h1_new-h1_old:+7.3f} {delta:+7.3f}")

    if delta < -args.tolerance:
        print(f"\nREFUSED: MRR dropped {abs(delta):.3f} > tolerance {args.tolerance:.3f}.")
        print(f"alias stays on {alias_targets(client)}; v2 is kept for investigation.")
        print("This gate is the entire point of building v2 in the dark.")
        return

    point_alias(client, "ch30_v2")
    print(f"\nACCEPTED: alias {ALIAS} -> {alias_targets(client)} (atomic switch)")
    print(f"storage: {dim_v1*4/1024:.1f} KB -> {dim_v2*4/1024:.1f} KB per vector, "
          f"{1_000_000*dim_v1*4/1e9:.1f} GB -> {1_000_000*dim_v2*4/1e9:.1f} GB per 1M points")

    point_alias(client, "ch30_v1")
    print(f"rollback drill: alias {ALIAS} -> {alias_targets(client)} again, "
          f"no re-indexing, no downtime")
    point_alias(client, "ch30_v2")
    print(f"and forward again -> {alias_targets(client)}")
    print("\nkeep v1 until the new one has survived a full traffic cycle, then drop it.")
    print("NOTE: in QDRANT_MODE=memory everything above is in-process; on a server the")
    print("alias flip is what makes this invisible to readers.")


if __name__ == "__main__":
    main()
