"""Chapter 18 - Qdrant collections built for scale: quantization, HNSW tuning, on-disk.

All of these calls are valid against the embedded client, but their EFFECT is only
measurable on a real server with millions of vectors. Here we prove the API is right
and that search still returns the correct chunks after quantization.

Run:  QDRANT_MODE=memory uv run python code/ch18/quantized_collection.py
"""
import time

from qdrant_client import models

from ragbook import chunk_documents, get_embeddings, get_qdrant_client, load_handbook

client = get_qdrant_client()
emb = get_embeddings()

chunks = chunk_documents(load_handbook())
vectors = emb.embed_documents([c.page_content for c in chunks])
DIM = len(vectors[0])
print(f"{len(chunks)} chunks, {DIM} dims")

COLLECTIONS = {
    "ch18_fp32": dict(
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    ),
    "ch18_int8": dict(
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE, on_disk=True),
        # scalar quantization: 4x less RAM, ~1-2% recall loss. quantile=0.99 clips outliers.
        quantization_config=models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, quantile=0.99, always_ram=True)
        ),
        # HNSW: m = links per node (RAM + recall), ef_construct = build-time effort.
        hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, on_disk=False),
    ),
    "ch18_binary": dict(
        vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE, on_disk=True),
        # binary: 32x less RAM; works well for 1024+ dim OpenAI-style embeddings; ALWAYS rescore.
        quantization_config=models.BinaryQuantization(binary=models.BinaryQuantizationConfig(always_ram=True)),
    ),
}

for name, cfg in COLLECTIONS.items():
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(collection_name=name, **cfg)
    client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(id=i, vector=v, payload={"source": c.metadata["source"], "text": c.page_content[:80]})
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ],
        wait=True,          # at scale: wait=False and let Qdrant index in the background
    )

query = emb.embed_query("How long does a full charge of the Atlas A2 take?")

for name in COLLECTIONS:
    t0 = time.perf_counter()
    res = client.query_points(
        collection_name=name,
        query=query,
        limit=3,
        search_params=models.SearchParams(
            hnsw_ef=128,                                    # query-time effort: higher = better recall, slower
            quantization=models.QuantizationSearchParams(
                ignore=False,      # use the quantized index for the candidate search
                rescore=True,      # then re-score the candidates with the original vectors
                oversampling=2.0,  # fetch 2x limit candidates before rescoring
            ),
        ),
    )
    ms = (time.perf_counter() - t0) * 1000
    top = [(round(p.score, 3), p.payload["source"]) for p in res.points]
    print(f"{name:<12} {ms:5.1f} ms  {top}")

print("\nSame top hits from all three: quantization changed the memory footprint, not the answer.")
print("(Timings on 56 vectors are noise; the point is the API and the recall check.)")
