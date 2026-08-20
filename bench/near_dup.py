"""Deterministic near-duplicate detection via MinHash + LSH banding.

Used to build the leakage-control group index: texts that are near-duplicates
(a human source and its machine continuation/rewrite, or the same passage
scraped into two cohorts) must share a group so they never land in different
train/development/calibration/test partitions.

Everything here is deterministic: the hash family is seeded, word hashes are
content-addressed, and union-find produces stable cluster labels. No raw text
is stored — only cluster assignments and summary statistics.

The implementation is dependency-free (numpy only) so the bench stays lean.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

# A large prime for the (a*x + b) mod p hash family (2^61 - 1, a Mersenne prime).
_MERSENNE = np.uint64((1 << 61) - 1)
_UINT64_MAX = np.uint64(0xFFFFFFFFFFFFFFFF)

_WORD_RE = re.compile(r"[a-z0-9']+")


def _word_hashes(text: str, ngram: int) -> np.ndarray:
    """Content-addressed 64-bit hashes of the word n-grams in `text`."""
    words = _WORD_RE.findall(text.lower())
    if not words:
        return np.empty(0, dtype=np.uint64)
    if ngram <= 1:
        shingles = words
    else:
        shingles = [" ".join(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
        if not shingles:
            shingles = words
    hashes = np.empty(len(shingles), dtype=np.uint64)
    for i, shingle in enumerate(shingles):
        digest = hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest()
        hashes[i] = np.frombuffer(digest, dtype=np.uint64)[0]
    # Unique shingles only: duplicates cannot change the per-permutation minimum.
    return np.unique(hashes)


def _hash_family(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.integers(1, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, np.iinfo(np.uint64).max, size=num_perm, dtype=np.uint64)
    return a, b


def minhash_signature(
    text: str,
    a: np.ndarray,
    b: np.ndarray,
    ngram: int = 3,
) -> np.ndarray:
    """MinHash signature of one document under the (a, b) hash family.

    signature[i] = min over shingles of (a[i]*shingle + b[i]) mod p. Computed
    with uint64 overflow; the Mersenne modulus keeps the family universal enough
    for near-duplicate detection.
    """
    shingles = _word_hashes(text, ngram)
    num_perm = len(a)
    if shingles.size == 0:
        return np.full(num_perm, _UINT64_MAX, dtype=np.uint64)
    # (num_perm, n_shingles) via outer product, then column-wise minimum.
    values = np.outer(a, shingles) + b[:, None]
    values = np.mod(values, _MERSENNE)
    return values.min(axis=1).astype(np.uint64)


def signatures_for(
    texts: list[str], num_perm: int = 128, ngram: int = 3, seed: int = 13
) -> np.ndarray:
    """(n_texts, num_perm) MinHash signature matrix, computed row by row."""
    a, b = _hash_family(num_perm, seed)
    out = np.empty((len(texts), num_perm), dtype=np.uint64)
    for i, text in enumerate(texts):
        out[i] = minhash_signature(text, a, b, ngram=ngram)
    return out


def lsh_candidate_pairs(signatures: np.ndarray, bands: int = 32) -> list[tuple[int, int]]:
    """Candidate near-duplicate pairs from LSH banding of the signatures.

    Two documents are candidates when they agree on every row of at least one
    band. With num_perm = bands * rows, the candidate threshold approximates the
    Jaccard similarity (1/bands)^(1/rows).
    """
    n, num_perm = signatures.shape
    if num_perm % bands != 0:
        raise ValueError("num_perm must be divisible by bands")
    rows = num_perm // bands
    # Star topology per bucket: union each document with the bucket's first
    # member. This yields the same connected components as all-pairs but emits
    # O(k) edges per bucket instead of O(k^2), which matters for large clusters.
    first_in_bucket: dict[tuple[int, bytes], int] = {}
    for band in range(bands):
        block = signatures[:, band * rows : (band + 1) * rows]
        for i in range(n):
            key = (band, block[i].tobytes())
            anchor = first_in_bucket.get(key)
            if anchor is None:
                first_in_bucket[key] = i
            elif anchor != i:
                yield (anchor, i)


def _union_find_clusters(n: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots: dict[int, int] = {}
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
        labels[i] = roots[root]
    return labels


def near_duplicate_clusters(
    texts: list[str],
    num_perm: int = 128,
    bands: int = 32,
    ngram: int = 3,
    seed: int = 13,
) -> tuple[np.ndarray, dict]:
    """Cluster texts into near-duplicate groups.

    Returns (labels, stats) where labels[i] is the cluster id of texts[i] and
    stats summarizes the clustering. Singletons each get their own cluster id.
    """
    n = len(texts)
    if n == 0:
        return np.empty(0, dtype=np.int64), {"n_texts": 0, "n_clusters": 0}
    signatures = signatures_for(texts, num_perm=num_perm, ngram=ngram, seed=seed)
    pairs = list(lsh_candidate_pairs(signatures, bands=bands))
    labels = _union_find_clusters(n, pairs)
    _, sizes = np.unique(labels, return_counts=True)
    stats = {
        "n_texts": int(n),
        "n_clusters": int(len(sizes)),
        "n_candidate_pairs": int(len(pairs)),
        "n_multi_document_clusters": int((sizes > 1).sum()),
        "largest_cluster": int(sizes.max()),
        "singletons": int((sizes == 1).sum()),
        "num_perm": int(num_perm),
        "bands": int(bands),
        "ngram": int(ngram),
        "seed": int(seed),
        "approx_jaccard_threshold": round((1.0 / bands) ** (1.0 / (num_perm // bands)), 3),
    }
    return labels, stats
