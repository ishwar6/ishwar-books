"""Chapter 30 - near-duplicate detection: MinHash, SimHash, and the LSH S-curve.

    uv run python code/ch30/dedup_minhash.py

At 1M documents you will have duplicates: the same policy attached to three
tickets, a v2 that changed one paragraph, the same press release on forty
mirrors. Duplicates are not just wasted embedding spend - they crowd the top-k
so one document occupies every slot the LLM was going to read.

Exact hashing catches byte-identical copies and nothing else. Change one comma
and the hash is unrelated. So you need a hash whose COLLISIONS ARE THE POINT:

    MinHash   estimates Jaccard overlap of the two shingle sets
    SimHash   estimates cosine-ish similarity by weighted bit voting
    LSH       makes it sub-quadratic: band the signature so only likely pairs
              are ever compared

No API calls: this is pure arithmetic over text.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re

import numpy as np

from ragbook import load_handbook

MASK64 = (1 << 64) - 1


# ------------------------------------------------------------- signatures ---
def shingles(text: str, k: int = 5) -> set[int]:
    """Hashed k-word shingles. Word-level shingles are robust to whitespace and
    case; k=5 is the usual compromise between sensitivity and false matches."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {
        int.from_bytes(hashlib.blake2b(" ".join(words[i:i + k]).encode(),
                                       digest_size=8).digest(), "big")
        for i in range(max(len(words) - k + 1, 1))
    }


def minhash(shingle_set: set[int], perms: np.ndarray) -> np.ndarray:
    """One signature row per permutation: min over h(shingle).

    P(min_h(A) == min_h(B)) == Jaccard(A, B) exactly - that identity is the
    whole algorithm, and the number of permutations only controls the variance
    of the estimate (std ~ 1/sqrt(n)).
    """
    x = np.fromiter(shingle_set, dtype=np.uint64, count=len(shingle_set))
    a, b = perms[:, 0:1], perms[:, 1:2]
    hashed = (a * x[None, :] + b) & np.uint64(MASK64)
    return hashed.min(axis=1)


def simhash(text: str, bits: int = 64) -> int:
    """Charikar SimHash: each feature votes +1/-1 on every bit, sign becomes the
    bit. Similar documents differ in few bits; Hamming distance is the metric."""
    votes = np.zeros(bits, dtype=np.int32)
    for shingle in shingles(text, k=3):
        for i in range(bits):
            votes[i] += 1 if (shingle >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if votes[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


# --------------------------------------------------------------- variants ---
def make_variants(text: str, rng: random.Random) -> dict[str, str]:
    """The duplicate shapes a real corpus actually contains."""
    words = text.split()

    def perturb(fraction: float) -> str:
        out = list(words)
        for i in rng.sample(range(len(out)), max(1, int(len(out) * fraction))):
            out[i] = rng.choice(["revised", "updated", "approximately", "the", "policy"])
        return " ".join(out)

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    shuffled = paragraphs[:]
    rng.shuffle(shuffled)

    return {
        "exact copy": text,
        "whitespace/case": re.sub(r"\s+", " ", text).upper(),
        "1% words changed": perturb(0.01),
        "5% words changed": perturb(0.05),
        "20% words changed": perturb(0.20),
        "paragraphs reordered": "\n\n".join(shuffled),
        "50% truncated": " ".join(words[: len(words) // 2]),
    }


def lsh_probability(jac: float, bands: int, rows: int) -> float:
    """P(at least one band matches) = 1 - (1 - s^r)^b - the S-curve that decides
    which pairs you ever look at. Its knee sits near (1/b)^(1/r)."""
    return 1 - (1 - jac ** rows) ** bands


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perms", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    perms = np.stack([
        np_rng.integers(1, MASK64, size=args.perms, dtype=np.uint64),
        np_rng.integers(0, MASK64, size=args.perms, dtype=np.uint64),
    ], axis=1)

    docs = load_handbook()
    base = docs[1].page_content                      # the PTO policy
    variants = make_variants(base, rng)

    base_sh, base_mh, base_si = shingles(base), None, simhash(base)
    base_mh = minhash(base_sh, perms)
    base_sha = hashlib.sha256(base.encode()).hexdigest()

    print(f"base document: {docs[1].metadata['source']} "
          f"({len(base.split())} words, {len(base_sh)} shingles), "
          f"{args.perms} permutations\n")
    print(f"{'variant':22} {'sha256':>7} {'true J':>7} {'MinHash J':>10} "
          f"{'err':>6} {'SimHash d':>10}")
    print("-" * 68)
    for name, text in variants.items():
        sh = shingles(text)
        mh = minhash(sh, perms)
        est = float((mh == base_mh).mean())
        true = jaccard(base_sh, sh)
        same_sha = hashlib.sha256(text.encode()).hexdigest() == base_sha
        print(f"{name:22} {'MATCH' if same_sha else '  -':>7} {true:7.3f} {est:10.3f} "
              f"{abs(est - true):6.3f} {hamming(base_si, simhash(text)):10}")

    # -- a different document, for the false-positive side of the ledger -----
    other = docs[5].page_content
    other_sh = shingles(other)
    print(f"{'UNRELATED doc':22} {'  -':>7} {jaccard(base_sh, other_sh):7.3f} "
          f"{float((minhash(other_sh, perms) == base_mh).mean()):10.3f} "
          f"{'':6} {hamming(base_si, simhash(other)):10}")

    # ------------------------------------------------ threshold selection --
    print("\nchoosing a threshold - every (document, variant) pair labelled:")
    pairs: list[tuple[float, bool]] = []
    for doc in docs:
        sh_a = shingles(doc.page_content)
        mh_a = minhash(sh_a, perms)
        for name, text in make_variants(doc.page_content, rng).items():
            if name == "50% truncated":              # deliberately borderline: not a dup
                continue
            est = float((minhash(shingles(text), perms) == mh_a).mean())
            pairs.append((est, True))
        for other_doc in docs:
            if other_doc.metadata["doc_id"] != doc.metadata["doc_id"]:
                est = float((minhash(shingles(other_doc.page_content), perms) == mh_a).mean())
                pairs.append((est, False))

    n_dup = sum(1 for _, is_dup in pairs if is_dup)
    print(f"  {len(pairs)} pairs, {n_dup} true duplicates, {len(pairs) - n_dup} distinct\n")
    print(f"{'threshold':>9} {'precision':>10} {'recall':>7} {'F1':>6}")
    best = (0.0, 0.0)
    for threshold in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        tp = sum(1 for s, d in pairs if s >= threshold and d)
        fp = sum(1 for s, d in pairs if s >= threshold and not d)
        fn = sum(1 for s, d in pairs if s < threshold and d)
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        best = max(best, (f1, threshold))
        print(f"{threshold:9.1f} {precision:10.2f} {recall:7.2f} {f1:6.2f}")
    print(f"  -> best F1 {best[0]:.2f} at threshold {best[1]:.1f}")

    # ------------------------------------------------------- the S-curve ---
    print(f"\nLSH banding with {args.perms} rows, the reason this is sub-quadratic:")
    print(f"{'bands x rows':>14} " + "".join(f"{j:>7.1f}" for j in (0.3, 0.5, 0.7, 0.8, 0.9)))
    print(f"{'(Jaccard ->)':>14}")
    for bands in (8, 16, 32):
        rows = args.perms // bands
        probs = "".join(f"{lsh_probability(j, bands, rows):7.2f}"
                        for j in (0.3, 0.5, 0.7, 0.8, 0.9))
        print(f"{f'{bands} x {rows}':>14} {probs}")
    print("  more bands -> catches weaker duplicates, compares more candidate pairs")


if __name__ == "__main__":
    main()
