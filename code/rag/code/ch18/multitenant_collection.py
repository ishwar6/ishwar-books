"""Chapter 18 - one collection, many customers (multitenancy) + zero-downtime re-embedding
with collection aliases.

Run:  QDRANT_MODE=memory uv run python code/ch18/multitenant_collection.py
"""
from qdrant_client import models

from ragbook import chunk_documents, get_embeddings, get_qdrant_client, load_handbook

client = get_qdrant_client()
emb = get_embeddings()
NAME = "ch18_tenants"

chunks = chunk_documents(load_handbook())
vectors = emb.embed_documents([c.page_content for c in chunks])
DIM = len(vectors[0])

if client.collection_exists(NAME):
    client.delete_collection(NAME)
client.create_collection(
    collection_name=NAME,
    vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE),
    # m=0 disables the GLOBAL HNSW graph; payload_m=16 builds one small graph PER tenant value.
    # A tenant-filtered search then walks only its own graph. Server-only effect.
    hnsw_config=models.HnswConfigDiff(m=0, payload_m=16),
)
# is_tenant=True tells Qdrant to co-locate each tenant's vectors on disk (fewer random reads).
client.create_payload_index(
    collection_name=NAME,
    field_name="tenant_id",
    field_schema=models.KeywordIndexParams(type=models.KeywordIndexType.KEYWORD, is_tenant=True),
)

# Pretend two customers uploaded the same handbook: odd chunks -> acme, even -> globex.
tenants = ["acme", "globex"]
client.upsert(
    collection_name=NAME,
    points=[
        models.PointStruct(id=i, vector=v, payload={"tenant_id": tenants[i % 2], "source": c.metadata["source"]})
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ],
)

q = emb.embed_query("hotel cap in Europe")
for tenant in tenants:
    res = client.query_points(
        collection_name=NAME,
        query=q,
        limit=3,
        query_filter=models.Filter(must=[models.FieldCondition(key="tenant_id", match=models.MatchValue(value=tenant))]),
    )
    print(f"{tenant:>7}: {[(p.id, p.payload['source']) for p in res.points]}")
print("Different ids per tenant: the filter is part of the search, not a post-filter.\n")

# --- Re-embedding without downtime: alias flips -------------------------------
# The app always reads from the alias "handbook_live". Build the new collection
# (new embedding model / new chunking) next to the old one, then flip the alias atomically.
for c in ("ch18_v1", "ch18_v2"):
    if client.collection_exists(c):
        client.delete_collection(c)
    client.create_collection(c, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE))

client.update_collection_aliases(
    change_aliases_operations=[models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name="ch18_v1", alias_name="handbook_live"))]
)
print("alias handbook_live ->", [a.collection_name for a in client.get_aliases().aliases if a.alias_name == "handbook_live"])

client.update_collection_aliases(
    change_aliases_operations=[
        models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name="handbook_live")),
        models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name="ch18_v2", alias_name="handbook_live")),
    ]
)
print("alias handbook_live ->", [a.collection_name for a in client.get_aliases().aliases if a.alias_name == "handbook_live"])
print("Readers never noticed. Delete ch18_v1 when you are sure.")
