"""Chapter 17 - the RAG pipeline as a FastAPI service.

Run:   QDRANT_MODE=memory uv run python code/ch17/app.py        (starts uvicorn on :8000)
Then:  uv run python code/ch17/client_demo.py

What matters here:
  * lifespan: build the expensive objects (Qdrant client, embeddings, LLM) ONCE
  * async endpoints that `await` LangChain's async methods -> the event loop stays free
  * auth + per-user RBAC filter, a small cache, timing and usage in every response
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qdrant_client import models

from ragbook import build_handbook_index, format_docs, get_llm

log = logging.getLogger("rag-api")
logging.basicConfig(level=logging.INFO, format='{"t":"%(asctime)s","lvl":"%(levelname)s","msg":%(message)s}')
for _name in ("httpx", "httpx2"):                          # openai>=3 ships its own httpx fork, logger "httpx2"
    logging.getLogger(_name).setLevel(logging.WARNING)    # one structured line per request, not per HTTP call

# ------------------------------------------------------------- auth/RBAC ------
# Demo API keys -> role. In production: JWT (verify signature, exp, issuer) or Okta/OIDC.
API_KEYS = {"demo-employee-key": "employee", "demo-sales-key": "sales"}
ROLE_DENY = {"employee": ["10-pricing-and-plans"], "sales": []}

CACHE: dict[str, tuple[float, dict]] = {}     # question hash -> (expires_at, response)
CACHE_TTL = 300


def current_role(x_api_key: str = Header(...)) -> str:
    """A FastAPI dependency: runs before the endpoint, rejects unknown keys."""
    role = API_KEYS.get(x_api_key)
    if not role:
        raise HTTPException(status_code=401, detail="invalid API key")
    return role


# --------------------------------------------------------------- schemas ------
class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int
    usage: dict
    cached: bool = False
    role: Literal["employee", "sales"]


# -------------------------------------------------------------- lifespan ------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: one Qdrant client, one embeddings client, one LLM - shared by all requests.
    app.state.store = build_handbook_index("ch17_handbook")
    app.state.llm = get_llm(timeout=30, max_retries=2)
    log.info(json.dumps("ready"))
    yield
    # Shutdown: close clients, flush logs. (Qdrant embedded mode releases its lock here.)
    app.state.store.client.close()


app = FastAPI(title="Lumora Handbook RAG", lifespan=lifespan)


def role_filter(role: str):
    denied = ROLE_DENY[role]
    if not denied:
        return None
    return models.Filter(must_not=[models.FieldCondition(key="metadata.doc_id", match=models.MatchAny(any=denied))])


PROMPT = ("Answer ONLY from the context, cite as [n]. If the answer is not in the context, say "
          "\"I don't know based on the handbook.\"\n\nContext:\n{context}\n\nQuestion: {question}")


# ------------------------------------------------------------- endpoints ------
@app.get("/health")
async def health():
    # Rule from 17.3: never call a sync SDK inside `async def`. The Qdrant client here is sync,
    # so hand the call to a worker thread instead of blocking the event loop.
    n = await asyncio.to_thread(lambda: app.state.store.client.count("ch17_handbook").count)
    return {"status": "ok", "points": n}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, role: str = Depends(current_role)):
    t0 = time.perf_counter()
    key = hashlib.sha256(f"{role}|{req.k}|{req.question.strip().lower()}".encode()).hexdigest()
    hit = CACHE.get(key)
    if hit and hit[0] > time.time():
        return {**hit[1], "cached": True, "latency_ms": int((time.perf_counter() - t0) * 1000)}

    # `await` = the event loop serves OTHER requests while we wait on Qdrant/OpenAI.
    docs = await app.state.store.asimilarity_search(req.question, k=req.k, filter=role_filter(role))
    try:
        msg = await asyncio.wait_for(
            app.state.llm.ainvoke(PROMPT.format(context=format_docs(docs), question=req.question)),
            timeout=40,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="LLM timeout")

    resp = {
        "answer": msg.text,
        "sources": sorted({d.metadata["source"] for d in docs}),
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "usage": msg.usage_metadata or {},
        "role": role,
    }
    CACHE[key] = (time.time() + CACHE_TTL, resp)
    log.info(json.dumps({"q": req.question, "role": role, "ms": resp["latency_ms"], "tokens": resp["usage"].get("total_tokens")}))
    return resp


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, role: str = Depends(current_role)):
    docs = await app.state.store.asimilarity_search(req.question, k=req.k, filter=role_filter(role))
    prompt = PROMPT.format(context=format_docs(docs), question=req.question)

    async def gen():
        async for chunk in app.state.llm.astream(prompt):     # tokens as they arrive
            if chunk.text:
                yield chunk.text
        yield "\n\n[sources: " + ", ".join(sorted({d.metadata["source"] for d in docs})) + "]\n"

    return StreamingResponse(gen(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), log_level="warning")
