"""Chapter 30 - the ingestion pipeline you actually run at scale, in one file.

    QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py
    QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py --fail-rate 0.3
    QDRANT_MODE=memory uv run python code/ch30/ingest_queue.py --workers 8 --repeat 3

A laptop script is `for doc in docs: index(doc)`. That loop has no answer to:
the process died at document 40,000; this PDF crashes the parser every time;
the embedding API returned 429; someone re-ran it and now everything is
duplicated. Those four questions are what turns a loop into a pipeline.

Everything here is deliberately small and local - a SQLite table is the work
queue - but every mechanism maps 1:1 onto SQS/Kafka/Celery:

    state machine   queued -> leased -> done
                              |  \\-> failed (retry with backoff) -> ... -> dead
    lease           a worker owns an item for N seconds; a crash just lets the
                    lease expire and someone else picks it up (at-least-once)
    idempotency     point ids are uuid5(doc_id, chunk_index) and work is keyed
                    by content hash, so re-delivery overwrites instead of
                    duplicating
    DLQ             `dead` rows: poison documents stop costing money and stay
                    visible for a human
"""
from __future__ import annotations

import argparse
import hashlib
import queue
import random
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from qdrant_client import models

from ragbook import DATA_DIR, chunk_documents, get_embeddings, get_qdrant_client, load_handbook
from langchain_core.documents import Document

DB_PATH = DATA_DIR / "ch30_queue.sqlite"
COLLECTION = "ch30_corpus"
NAMESPACE = uuid.UUID("6f1c2a3e-0000-4000-8000-0000000030a9")
MAX_ATTEMPTS = 3
LEASE_SECONDS = 30


# ------------------------------------------------------------------ queue ---
class WorkQueue:
    """A durable work queue in SQLite. WAL + a lock makes it safe across threads."""

    def __init__(self, path: Path, reset: bool = False):
        if reset and path.exists():
            path.unlink()
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS items (
                doc_id       TEXT PRIMARY KEY,
                path         TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                state        TEXT NOT NULL DEFAULT 'queued',
                attempts     INTEGER NOT NULL DEFAULT 0,
                lease_until  REAL NOT NULL DEFAULT 0,
                next_try     REAL NOT NULL DEFAULT 0,
                last_error   TEXT,
                chunks       INTEGER NOT NULL DEFAULT 0,
                updated_at   REAL NOT NULL DEFAULT 0)""")
        self.db.commit()

    def submit(self, doc_id: str, path: str, content_hash: str) -> str:
        """Returns what happened: 'new', 'changed' or 'unchanged'.

        This is the incremental-indexing decision, and it is the whole reason a
        second run of a 1M-document crawl costs nothing.
        """
        with self.lock:
            row = self.db.execute(
                "SELECT content_hash, state FROM items WHERE doc_id=?", (doc_id,)).fetchone()
            if row is None:
                self.db.execute(
                    "INSERT INTO items(doc_id, path, content_hash, updated_at) VALUES (?,?,?,?)",
                    (doc_id, path, content_hash, time.time()))
                self.db.commit()
                return "new"
            if row[0] == content_hash and row[1] == "done":
                return "unchanged"
            self.db.execute(
                "UPDATE items SET content_hash=?, state='queued', attempts=0, "
                "next_try=0, last_error=NULL, updated_at=? WHERE doc_id=?",
                (content_hash, time.time(), doc_id))
            self.db.commit()
            return "changed"

    def lease(self) -> tuple[str, str] | None:
        """Atomically take one ready item. Expired leases are reclaimed here -
        that is how a crashed worker's work comes back without a supervisor."""
        now = time.time()
        with self.lock:
            row = self.db.execute(
                "SELECT doc_id, path FROM items "
                "WHERE (state='queued' AND next_try<=?) OR (state='leased' AND lease_until<?) "
                "ORDER BY next_try LIMIT 1", (now, now)).fetchone()
            if row is None:
                return None
            self.db.execute(
                "UPDATE items SET state='leased', lease_until=?, attempts=attempts+1 "
                "WHERE doc_id=?", (now + LEASE_SECONDS, row[0]))
            self.db.commit()
            return row[0], row[1]

    def complete(self, doc_id: str, n_chunks: int) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE items SET state='done', chunks=?, last_error=NULL, updated_at=? "
                "WHERE doc_id=?", (n_chunks, time.time(), doc_id))
            self.db.commit()

    def fail(self, doc_id: str, error: str) -> str:
        """Retryable until MAX_ATTEMPTS, then the dead-letter queue.

        Backoff is exponential WITH JITTER: without jitter every worker retries
        the same failing dependency at the same instant and you rebuild the
        stampede you were trying to avoid.
        """
        with self.lock:
            attempts = self.db.execute(
                "SELECT attempts FROM items WHERE doc_id=?", (doc_id,)).fetchone()[0]
            if attempts >= MAX_ATTEMPTS:
                state, delay = "dead", 0.0
            else:
                state = "queued"
                delay = (2 ** attempts) * 0.05 * (0.5 + random.random())
            self.db.execute(
                "UPDATE items SET state=?, next_try=?, last_error=?, updated_at=? WHERE doc_id=?",
                (state, time.time() + delay, error[:200], time.time(), doc_id))
            self.db.commit()
            return state

    def counts(self) -> dict[str, int]:
        with self.lock:
            return dict(self.db.execute(
                "SELECT state, COUNT(*) FROM items GROUP BY state").fetchall())

    def pending(self) -> int:
        c = self.counts()
        return c.get("queued", 0) + c.get("leased", 0)

    def dead_letters(self) -> list[tuple[str, str]]:
        with self.lock:
            return self.db.execute(
                "SELECT doc_id, last_error FROM items WHERE state='dead'").fetchall()


# ------------------------------------------------------------------ stats ---
@dataclass
class Stats:
    parsed: int = 0
    embedded_chunks: int = 0
    retries: int = 0
    dead: int = 0
    parse_s: float = 0.0
    embed_s: float = 0.0
    upsert_s: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------- workers ---
def source_documents(repeat: int) -> list[Document]:
    """The corpus: the handbook plus anything Chapter 7 generated. `--repeat`
    clones it with fresh ids to make the worker-scaling numbers visible."""
    docs = list(load_handbook())
    gen = DATA_DIR / "generated"
    if gen.exists():
        for path in sorted(gen.glob("*.md")):
            docs.append(Document(page_content=path.read_text(encoding="utf-8"),
                                 metadata={"source": path.name, "doc_id": path.stem,
                                           "title": path.stem}))
    if repeat > 1:
        clones = []
        for copy in range(repeat):
            for d in docs:
                meta = dict(d.metadata, doc_id=f"{d.metadata['doc_id']}__c{copy}")
                clones.append(Document(page_content=d.page_content, metadata=meta))
        docs = clones
    return docs


def worker(name: str, q: WorkQueue, corpus: dict[str, Document], out: queue.Queue,
           emb, stats: Stats, fail_rate: float, poison: set[str],
           stop: threading.Event) -> None:
    while not stop.is_set():
        item = q.lease()
        if item is None:
            if q.pending() == 0:
                return
            time.sleep(0.02)
            continue
        doc_id, _ = item
        try:
            t0 = time.perf_counter()
            if doc_id in poison:
                # A PERMANENT failure: this document will never parse, no matter
                # how often you retry. Retrying it forever is how a pipeline
                # burns a budget on one corrupt file; the DLQ is the answer.
                raise ValueError("unsupported encrypted PDF (permanent)")
            if random.random() < fail_rate:
                # Stands in for the real ones: a corrupt PDF, a 500 from object
                # storage, an OOM in the parser. Transient vs permanent is the
                # only distinction the queue needs.
                raise RuntimeError("parser crashed on page 3")
            doc = corpus[doc_id]
            chunks = chunk_documents([doc])
            t1 = time.perf_counter()

            vectors = emb.embed_documents([c.page_content for c in chunks])
            t2 = time.perf_counter()

            out.put((doc_id, chunks, vectors))
            with stats.lock:
                stats.parsed += 1
                stats.embedded_chunks += len(chunks)
                stats.parse_s += t1 - t0
                stats.embed_s += t2 - t1
        except Exception as exc:                      # noqa: BLE001 - the point is to catch all
            state = q.fail(doc_id, f"{type(exc).__name__}: {exc}")
            with stats.lock:
                if state == "dead":
                    stats.dead += 1
                else:
                    stats.retries += 1


def writer(q: WorkQueue, out: queue.Queue, store_client, stats: Stats,
           stop: threading.Event) -> None:
    """One writer owns the index. Workers are parallel; writes are batched and
    serial, which keeps the vector DB happy and makes upserts idempotent."""
    while not (stop.is_set() and out.empty()):
        try:
            doc_id, chunks, vectors = out.get(timeout=0.05)
        except queue.Empty:
            continue
        t0 = time.perf_counter()
        points = [
            models.PointStruct(
                id=str(uuid.uuid5(NAMESPACE, f"{doc_id}::{i}")),   # stable => idempotent
                vector=vec,
                payload={"page_content": c.page_content,
                         "metadata": {**c.metadata, "doc_id": doc_id}})
            for i, (c, vec) in enumerate(zip(chunks, vectors))
        ]
        store_client.upsert(COLLECTION, points=points, wait=False)
        q.complete(doc_id, len(chunks))
        with stats.lock:
            stats.upsert_s += time.perf_counter() - t0


# ------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fail-rate", type=float, default=0.0,
                    help="probability a document fails parsing, to exercise retries/DLQ")
    ap.add_argument("--repeat", type=int, default=1, help="clone the corpus N times")
    ap.add_argument("--poison", type=int, default=0,
                    help="documents that fail PERMANENTLY, to exercise the DLQ")
    ap.add_argument("--reset", action="store_true", help="start from an empty queue")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    q = WorkQueue(DB_PATH, reset=args.reset)
    docs = source_documents(args.repeat)
    corpus = {d.metadata["doc_id"]: d for d in docs}

    submitted = {"new": 0, "changed": 0, "unchanged": 0}
    for d in docs:
        digest = hashlib.sha256(d.page_content.encode()).hexdigest()
        submitted[q.submit(d.metadata["doc_id"], d.metadata["source"], digest)] += 1
    print(f"submitted {len(docs)} documents: {submitted['new']} new, "
          f"{submitted['changed']} changed, {submitted['unchanged']} unchanged (skipped)")
    if q.pending() == 0:
        print("nothing to do - this is what a re-run of a finished crawl costs.")
        print(f"queue states: {q.counts()}")
        return

    client = get_qdrant_client()
    emb = get_embeddings()
    if not client.collection_exists(COLLECTION):
        client.create_collection(COLLECTION, vectors_config=models.VectorParams(
            size=len(emb.embed_query("probe")), distance=models.Distance.COSINE))

    poison = set(sorted(corpus)[: args.poison])
    if poison:
        print(f"poison documents (always fail): {', '.join(sorted(poison))}")

    stats, out, stop = Stats(), queue.Queue(), threading.Event()
    t_start = time.perf_counter()
    writer_thread = threading.Thread(
        target=writer, args=(q, out, client, stats, stop), daemon=True)
    writer_thread.start()
    threads = [threading.Thread(target=worker,
                                args=(f"w{i}", q, corpus, out, emb, stats,
                                      args.fail_rate, poison, stop),
                                daemon=True)
               for i in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    stop.set()
    writer_thread.join(timeout=10)
    elapsed = time.perf_counter() - t_start

    counts = q.counts()
    print(f"\nfinished in {elapsed:.1f}s with {args.workers} workers "
          f"(fail-rate {args.fail_rate:.0%})")
    print(f"  states        {counts}")
    print(f"  documents     {stats.parsed} indexed, {stats.retries} retries, "
          f"{stats.dead} dead-lettered")
    print(f"  chunks        {stats.embedded_chunks} embedded -> "
          f"{client.count(COLLECTION).count} points in Qdrant")
    print(f"  throughput    {stats.parsed / elapsed:.1f} docs/s, "
          f"{stats.embedded_chunks / elapsed:.1f} chunks/s")
    print(f"  worker time   parse {stats.parse_s:.1f}s | embed {stats.embed_s:.1f}s "
          f"| upsert {stats.upsert_s:.1f}s  (summed across workers)")

    dead = q.dead_letters()
    if dead:
        print(f"\ndead-letter queue ({len(dead)} documents - a human looks at these):")
        for doc_id, err in dead[:5]:
            print(f"  {doc_id:38} {err}")
    print(f"\nre-run the same command: unchanged documents are skipped, and the "
          f"point count stays at {client.count(COLLECTION).count} because ids are stable.")


if __name__ == "__main__":
    main()
