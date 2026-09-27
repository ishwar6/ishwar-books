"""Chapter 27 - diagnose.py: the rank-17 playbook, as a program.

Somebody reports "the correct document comes back at rank 17". Raising k to 20
makes the ticket go away and makes every future query slower, noisier and more
expensive. This script instead runs TWELVE diagnostics against one
(question, expected document) pair, prints a verdict table, and ends with a
ranked recommendation in which every suggestion is backed by a measurement it
just made.

  uv run python code/ch27/diagnose.py --golden q17
  uv run python code/ch27/diagnose.py --question "how fast is the robot" \
        --expect 04-atlas-a2-specification --keywords 2.0 speed

Nothing is hard-coded per question: the same twelve checks run for any input.
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import numpy as np
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ragbook import chunk_documents, get_embeddings, get_llm, load_golden, load_handbook

CANDIDATES = 20          # what a real first stage would hand to a re-ranker
TOP_K = 5                # what the LLM would actually read


# --------------------------------------------------------------- the world ---
class World:
    """Everything the diagnostics need, built once: chunks, their vectors, a
    dense ranking function, a BM25 ranking function."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120) -> None:
        self.chunks = chunk_documents(load_handbook(), chunk_size, chunk_overlap)
        self.ids = [c.metadata["chunk_id"] for c in self.chunks]
        self.by_id = {c.metadata["chunk_id"]: c for c in self.chunks}

        emb = get_embeddings()
        self.embed_query = emb.embed_query
        vecs = np.array(emb.embed_documents([c.page_content for c in self.chunks]), dtype=np.float32)
        self.matrix = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)   # unit vectors -> dot == cosine

        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([tokenize(c.page_content) for c in self.chunks])

    # Exact nearest neighbours over every chunk. The embedded Qdrant used in this
    # book is ALSO exact, so "ANN vs exact" only diverges against a server running
    # HNSW (diagnostic 6 explains what to do there).
    def dense_rank(self, text: str) -> list[tuple[str, float]]:
        q = np.array(self.embed_query(text), dtype=np.float32)
        q /= np.linalg.norm(q)
        sims = self.matrix @ q
        order = np.argsort(-sims)
        return [(self.ids[i], float(sims[i])) for i in order]

    def bm25_rank(self, text: str) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(text))
        order = np.argsort(-scores)
        return [(self.ids[i], float(scores[i])) for i in order]


def tokenize(text: str) -> list[str]:
    """Keep identifiers like `beacon-4187` and versions like `3.8` in one piece -
    they are exactly the tokens where lexical search beats dense search."""
    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())


def rank_of(ranking: list[tuple[str, float]], targets: set[str]) -> int | None:
    """1-based rank of the first target id in a ranking, or None."""
    for i, (cid, _) in enumerate(ranking, start=1):
        if cid in targets:
            return i
    return None


def rrf(rankings: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (cid, _) in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


# -------------------------------------------------------------- the report ---
class Check:
    def __init__(self, n: int, name: str, verdict: str, detail: str, fix: tuple[str, str] | None = None):
        self.n, self.name, self.verdict, self.detail = n, name, verdict, detail
        self.fix = fix                      # (cost tag, recommendation) or None


COST_ORDER = {"free": 0, "cheap": 1, "medium": 2, "expensive": 3}


def diagnose(world: World, question: str, expect: str, keywords: list[str]) -> list[Check]:
    checks: list[Check] = []

    doc_chunks = {c.metadata["chunk_id"] for c in world.chunks if c.metadata["doc_id"] == expect}
    kw_lower = [k.lower() for k in keywords]

    def contains_all(c: Document) -> bool:
        body = c.page_content.lower()
        return bool(kw_lower) and all(k in body for k in kw_lower)

    def contains_any(c: Document) -> bool:
        body = c.page_content.lower()
        return any(k in body for k in kw_lower)

    answer_chunks = {c.metadata["chunk_id"] for c in world.chunks if contains_all(c)}
    # A keyword like "No" matches half the corpus. If the "answer chunk" set is that
    # big the label is useless, so drop it rather than trust it - a lesson in its own
    # right: a diagnostic is only as good as the ground truth you feed it.
    vague_keywords = len(answer_chunks) > max(3, 0.25 * len(world.chunks))
    if vague_keywords:
        answer_chunks = set()
    partial_chunks = {c.metadata["chunk_id"] for c in world.chunks if contains_any(c)}
    # The chunks we actually want back: the ones holding the answer if we can
    # identify them, otherwise any chunk of the expected document.
    gold = answer_chunks or doc_chunks

    dense = world.dense_rank(question)
    dense_pos = {cid: i for i, (cid, _) in enumerate(dense, start=1)}
    doc_rank = rank_of(dense, doc_chunks)
    gold_rank = rank_of(dense, gold)

    # 1 -------------------------------------------------- is it even indexed?
    if not doc_chunks:
        checks.append(Check(1, "document is in the index", "FAIL",
                            f"no chunk has doc_id={expect!r} - ingestion or filtering dropped it",
                            ("cheap", "Fix ingestion first: nothing downstream can retrieve a document that is not there.")))
        return checks
    checks.append(Check(1, "document is in the index", "OK", f"{len(doc_chunks)} chunks indexed"))

    # Diagnostics 8 and 9 look at ONE gold chunk. Pick the best-ranked one rather
    # than an arbitrary member of a set - set iteration order varies between
    # processes, and a diagnostic that reports a different chunk on every run is
    # not a diagnostic.
    best_gold = next(cid for cid, _ in dense if cid in gold)

    # 2 ------------------------------------------- is the answer in ONE chunk?
    if not keywords:
        checks.append(Check(2, "answer lives in one chunk", "INFO", "no keywords given; cannot locate the answer chunk"))
    elif vague_keywords:
        checks.append(Check(2, "answer lives in one chunk", "INFO",
                            f"keywords {keywords} match a quarter of the corpus - too generic to locate the "
                            f"answer chunk; pass better ones with --keywords"))
    elif answer_chunks:
        checks.append(Check(2, "answer lives in one chunk", "OK",
                            f"{len(answer_chunks)} chunk(s) contain every keyword"))
    else:
        holders = sorted({world.by_id[c].metadata["chunk_index"] for c in partial_chunks})
        checks.append(Check(2, "answer lives in one chunk", "FAIL",
                            f"no single chunk has all of {keywords}; fragments in chunk_index {holders}",
                            ("expensive", "Re-chunk: the answer straddles a boundary. Bigger chunks, more overlap, "
                                          "or parent/child retrieval (Ch 9) so the reader gets the whole section.")))

    # 3 --------------------------------------------- where does it land, and why
    top_id, top_score = dense[0]
    gold_score = next(s for cid, s in dense if cid in gold)
    gap = top_score - gold_score
    verdict = "OK" if (gold_rank or 99) <= TOP_K else "FAIL"
    checks.append(Check(3, "rank / score gap (dense)", verdict,
                        f"answer chunk at rank {gold_rank} (score {gold_score:.3f}); "
                        f"top hit {world.by_id[top_id].metadata['source']} {top_score:.3f} (gap {gap:.3f})"))

    # 4 ------------------------------------------------- vocabulary mismatch
    q_terms = set(tokenize(question)) - STOPWORDS
    gold_text = " ".join(world.by_id[c].page_content for c in gold)
    g_terms = set(tokenize(gold_text))
    overlap = q_terms & g_terms
    ratio = len(overlap) / max(1, len(q_terms))
    if ratio < 0.5:
        checks.append(Check(4, "query/chunk vocabulary overlap", "FAIL",
                            f"{len(overlap)}/{len(q_terms)} query terms appear in the answer chunk "
                            f"(missing: {sorted(q_terms - g_terms)[:6]})",
                            ("cheap", "Vocabulary gap: the chunk never uses the asker's words. HyDE or a rewrite "
                                      "bridges it; contextual retrieval fixes it at ingest.")))
    else:
        checks.append(Check(4, "query/chunk vocabulary overlap", "OK",
                            f"{len(overlap)}/{len(q_terms)} query terms present ({ratio:.0%})"))

    # 5 --------------------------------------------------- would BM25 do better?
    bm = world.bm25_rank(question)
    bm_rank = rank_of(bm, gold)
    hybrid_rank = rank_of([(cid, s) for cid, s in rrf([dense, bm])], gold)
    better = min(x for x in (bm_rank or 10**6, hybrid_rank or 10**6))
    if better < (gold_rank or 10**6):
        checks.append(Check(5, "lexical / hybrid ranks it higher", "FAIL",
                            f"dense {gold_rank} → BM25 {bm_rank}, hybrid(RRF) {hybrid_rank}",
                            ("cheap", "Turn on hybrid search (Ch 8/23): the terms are IN the chunk, the embedding "
                                      "just does not weight them.")))
    else:
        checks.append(Check(5, "lexical / hybrid ranks it higher", "OK",
                            f"dense {gold_rank}, BM25 {bm_rank}, hybrid {hybrid_rank} - no lexical win"))

    # 6 ------------------------------------------------------- ANN vs exact
    #  This search is exact (numpy, and the embedded Qdrant too). On a server the
    #  same query goes through HNSW, where a low ef_search or aggressive
    #  quantization can drop the chunk out of the candidate list entirely.
    checks.append(Check(6, "ANN recall vs exact search", "INFO",
                        "this run is exact, so rank is a RANKING fault, not an index fault; "
                        "against a server, re-run with ef_search=512 / rescore=on to rule the index out"))

    # 7 ----------------------------------------------------------- filters
    meta_keys = sorted(set(world.by_id[best_gold].metadata) - {"chunk_id", "chunk_index"})
    checks.append(Check(7, "metadata filters exclude it", "INFO",
                        f"no filter applied in this run; fields available to filter on: {meta_keys}"))

    # 8 --------------------------------------------------- near-duplicate crowding
    gold_vec = world.matrix[world.ids.index(best_gold)]
    above = [cid for cid, _ in dense[: (gold_rank or 1) - 1]]
    dupes = [(cid, float(world.matrix[world.ids.index(cid)] @ gold_vec)) for cid in above]
    dupes = [(c, s) for c, s in dupes if s > 0.90]
    if dupes:
        checks.append(Check(8, "near-duplicates crowd it out", "FAIL",
                            f"{len(dupes)} chunk(s) above it are ≥0.90 similar to it",
                            ("cheap", "Deduplicate at ingest or use MMR/diversity at query time (Ch 29).")))
    else:
        checks.append(Check(8, "near-duplicates crowd it out", "OK", f"{len(above)} chunks rank above it, none ≥0.90 similar"))

    # 9 ---------------------------------------------- is the chunk itself weak?
    gc = world.by_id[best_gold]
    words = len(gc.page_content.split())
    lines = [l for l in gc.page_content.splitlines() if l.strip()]
    struct = sum(1 for l in lines if l.lstrip().startswith(("#", "|", "-", "*"))) / max(1, len(lines))
    if words < 25:
        checks.append(Check(9, "chunk is a usable unit", "FAIL", f"only {words} words - too small to embed meaningfully",
                            ("expensive", "Re-chunk larger, or index children but return parents (Ch 9).")))
    elif struct > 0.8:
        checks.append(Check(9, "chunk is a usable unit", "WARN",
                            f"{struct:.0%} of lines are headings/table syntax - the vector encodes layout, not meaning",
                            ("medium", "Prepend the heading path to the text, or add a one-line LLM-written summary "
                                       "of the chunk before embedding (contextual retrieval).")))
    else:
        checks.append(Check(9, "chunk is a usable unit", "OK", f"{words} words, {struct:.0%} structural lines"))

    # 10 ------------------------------------------------------- multi-hop?
    if keywords and not answer_chunks and len(partial_chunks) > 1:
        checks.append(Check(10, "single-hop question", "FAIL",
                            "the keywords live in different chunks - one retrieval cannot satisfy this question",
                            ("cheap", "Decompose the question (Ch 9/27.4) or let an agent retrieve twice (Ch 15).")))
    else:
        checks.append(Check(10, "single-hop question", "OK", "one chunk can answer it"))

    # 11 ------------------------------------------------ does HyDE move it?
    t0 = time.perf_counter()
    fake = (ChatPromptTemplate.from_messages([
        ("system", "Write two sentences from an internal company handbook that would answer the question. "
                   "Invent specifics; this text is only a search probe."),
        ("human", "{q}")]) | get_llm()).invoke({"q": question}).text
    hyde_rank = rank_of(world.dense_rank(fake), gold)
    dt = (time.perf_counter() - t0) * 1000
    if hyde_rank and gold_rank and hyde_rank < gold_rank:
        checks.append(Check(11, "HyDE moves it up", "FAIL",
                            f"rank {gold_rank} → {hyde_rank} with a hypothetical answer ({dt:.0f} ms)",
                            ("cheap", "Add HyDE for this class of question - confirms a question/statement phrasing gap.")))
    else:
        checks.append(Check(11, "HyDE moves it up", "OK", f"rank {gold_rank} → {hyde_rank} - rewriting does not help"))

    # 12 ------------------------------------- is it retrievable but mis-ranked?
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        ce = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
        cands = [cid for cid, _ in dense[:CANDIDATES]]
        scores = list(ce.rerank(question, [world.by_id[c].page_content for c in cands]))
        reranked = [c for c, _ in sorted(zip(cands, scores), key=lambda x: -x[1])]
        ce_rank = next((i for i, c in enumerate(reranked, start=1) if c in gold), None)
        if gold_rank and gold_rank > TOP_K and ce_rank and ce_rank <= TOP_K:
            checks.append(Check(12, "reranker rescues it", "FAIL",
                                f"in the top-{CANDIDATES} candidates and a cross-encoder lifts it {gold_rank} → {ce_rank}",
                                ("medium", "Retrieve wide (k=20–50) and re-rank to 5. The evidence is being found, "
                                           "only mis-ordered - this is exactly what re-ranking is for.")))
        elif ce_rank and gold_rank and ce_rank > gold_rank:
            checks.append(Check(12, "reranker rescues it", "WARN",
                                f"cross-encoder DEMOTES it {gold_rank} → {ce_rank}: the reranker shares the "
                                f"bi-encoder's blind spot here, so re-ranking is not the fix"))
        else:
            checks.append(Check(12, "reranker rescues it", "OK",
                                f"cross-encoder puts it at {ce_rank} of the top-{CANDIDATES} candidates"))
    except Exception as exc:                                   # offline / model not cached
        checks.append(Check(12, "reranker rescues it", "INFO", f"cross-encoder unavailable ({type(exc).__name__})"))

    return checks


STOPWORDS = {"the", "a", "an", "is", "are", "do", "does", "did", "what", "which", "how", "many",
             "much", "i", "we", "you", "of", "for", "to", "in", "on", "and", "or", "can", "my",
             "it", "its", "be", "use", "used", "uses", "get", "there", "that", "this", "at", "by"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose one retrieval failure.")
    ap.add_argument("--golden", help="golden question id, e.g. q17")
    ap.add_argument("--question")
    ap.add_argument("--expect", help="doc_id that should be retrieved")
    ap.add_argument("--keywords", nargs="*", default=[])
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    args = ap.parse_args()

    if args.golden:
        g = {x["id"]: x for x in load_golden()}[args.golden]
        question, expect = g["question"], g["sources"][0]
        keywords = args.keywords or g["keywords"]      # --keywords overrides weak golden labels
    elif args.question and args.expect:
        question, expect, keywords = args.question, args.expect, args.keywords
    else:
        sys.exit("give --golden q17, or --question ... --expect <doc_id> [--keywords ...]")

    print("=" * 78)
    print(" RANK-17 PLAYBOOK - one question, twelve diagnostics")
    print("=" * 78)
    print(f"question : {question}")
    print(f"expects  : {expect}    keywords: {keywords or '(none)'}")
    print(f"chunking : size={args.chunk_size} overlap={args.chunk_overlap}\n")

    world = World(args.chunk_size, args.chunk_overlap)
    dense = world.dense_rank(question)
    print(f"BASELINE dense top-{TOP_K}:")
    for i, (cid, s) in enumerate(dense[:TOP_K], start=1):
        c = world.by_id[cid]
        head = c.page_content.strip().splitlines()[0][:52]
        print(f"   {i}. {s:.3f}  {c.metadata['source']:<34} {head!r}")

    doc_chunks = {c.metadata["chunk_id"] for c in world.chunks if c.metadata["doc_id"] == expect}
    kw = [k.lower() for k in keywords]
    ans_chunks = {c.metadata["chunk_id"] for c in world.chunks
                  if kw and all(k in c.page_content.lower() for k in kw)}
    print(f"\n   rank of the first chunk from {expect}: {rank_of(dense, doc_chunks)}")
    if ans_chunks:
        ar = rank_of(dense, ans_chunks)
        print(f"   rank of the chunk that CONTAINS the answer : {ar}      <- the number that matters")
        if ar and ar > TOP_K:
            print(f"   document-level recall@{TOP_K} scores this retrieval 1.0. It failed.")

    checks = diagnose(world, question, expect, keywords)

    print(f"\n{'#':<3}{'check':<34}{'verdict':<9}detail")
    print("-" * 78)
    for c in checks:
        print(f"{c.n:<3}{c.name:<34}{c.verdict:<9}{c.detail}")

    fixes = sorted([c for c in checks if c.fix], key=lambda c: COST_ORDER[c.fix[0]])
    print("\nRECOMMENDATION (cheapest first; each line is backed by a check above)")
    if not fixes:
        print("   nothing is broken: the answer chunk is already in the top-5.")
    for c in fixes:
        print(f"   [{c.fix[0]:<9}] (from #{c.n}) {c.fix[1]}")
    print("\nNote: change ONE thing, then re-run this script and the Chapter 10 eval.")


if __name__ == "__main__":
    main()
