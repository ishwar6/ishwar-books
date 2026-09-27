"""Chapter 17 - exercise the running API. Start app.py first.

Run:  uv run python code/ch17/client_demo.py
"""
import asyncio
import time

import httpx

BASE = "http://127.0.0.1:8000"


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE, timeout=60) as c:
        print("health:", (await c.get("/health")).json())

        # missing header -> 422 (FastAPI validation); wrong key -> 401 (our dependency)
        print("\nmissing key ->", (await c.post("/ask", json={"question": "How many PTO days?"})).status_code)
        print("wrong key   ->", (await c.post("/ask", json={"question": "How many PTO days?"}, headers={"x-api-key": "nope"})).status_code)

        q = {"question": "How much does an Atlas A2 cost?"}
        for key in ("demo-employee-key", "demo-sales-key"):
            r = (await c.post("/ask", json=q, headers={"x-api-key": key})).json()
            print(f"\n{r['role']:>8}: {r['answer'][:100]}  ({r['latency_ms']} ms, {r['usage'].get('total_tokens')} tok)")

        r = (await c.post("/ask", json=q, headers={"x-api-key": "demo-sales-key"})).json()
        print(f"\ncached repeat: cached={r['cached']} latency={r['latency_ms']} ms")

        # 5 concurrent requests: with async endpoints they overlap instead of queueing.
        qs = ["hotel cap in Europe?", "Beacon API rate limit?", "Atlas A2 payload?", "probation length?", "RTO and RPO?"]
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[c.post("/ask", json={"question": x}, headers={"x-api-key": "demo-sales-key"}) for x in qs])
        wall = time.perf_counter() - t0
        per = [r.json()["latency_ms"] for r in rs]
        print(f"\n5 concurrent: wall={wall*1000:.0f} ms, sum of individual latencies={sum(per)} ms")

        print("\nstreaming: ", end="", flush=True)
        async with c.stream("POST", "/ask/stream", json={"question": "What are Beacon's RTO and RPO?"},
                            headers={"x-api-key": "demo-sales-key"}) as s:
            async for chunk in s.aiter_text():
                print(chunk, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
