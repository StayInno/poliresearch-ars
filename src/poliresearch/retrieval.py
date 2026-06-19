"""Retrieval — the long-term-memory read path (Karpathy's "disk" in the LLM-OS).

Ships BM25 (Okapi) as the default: term-weighted, length-normalised lexical ranking — a real
upgrade over raw keyword counts (critique #8), with zero dependencies and no model download. An
embedding backend implements the same `Retriever` interface and drops in when available, so the
generator / falsifier / discovery loop get better retrieval without any change.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


class Retriever(Protocol):
    def search(self, query: str, k: int = 6): ...


class BM25Retriever:
    def __init__(self, chunks, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [_tok(c.text) for c in chunks]
        self.N = len(self.docs)
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) if self.N else 0.0
        df: Counter = Counter()
        for d in self.docs:
            for t in set(d):
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self._tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int = 6):
        q = _tok(query)
        scored = []
        for i, tf in enumerate(self._tf):
            dl = len(self.docs[i])
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1))
                s += self.idf.get(t, 0.0) * (f * (self.k1 + 1)) / denom
            if s > 0:
                scored.append((s, self.chunks[i]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:k]]


class EmbeddingRetriever:
    """Optional dense retriever. Requires `sentence-transformers`; falls back to BM25 if absent."""

    def __init__(self, chunks, model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        self._fallback = None
        try:
            from sentence_transformers import SentenceTransformer, util  # type: ignore
            self._model = SentenceTransformer(model_name)
            self._util = util
            self._emb = self._model.encode([c.text for c in chunks], convert_to_tensor=True)
        except Exception:
            self._model = None
            self._fallback = BM25Retriever(chunks)

    def search(self, query: str, k: int = 6):
        if self._model is None:
            return self._fallback.search(query, k)
        qe = self._model.encode(query, convert_to_tensor=True)
        hits = self._util.semantic_search(qe, self._emb, top_k=k)[0]
        return [self.chunks[h["corpus_id"]] for h in hits]


def make_retriever(chunks, kind: str = "bm25") -> Retriever:
    return EmbeddingRetriever(chunks) if kind == "embedding" else BM25Retriever(chunks)
