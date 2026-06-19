"""Closed corpus: the only knowledge the system is allowed to answer from (deck slides 11-12).

This enforces the central physical contradiction's resolution by *separation on condition*:
factual answers must come from here, never from the model's parametric memory.

The corpus hash (gate 6) makes every run reproducible — change a single source file and the
hash changes, so an answer can always be tied to the exact corpus that produced it (DVC role).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

# Plain-text-extractable sources we index out of the box. PDFs are supported when a
# text-extraction backend is installed, but a closed corpus of .txt/.md works with no deps.
_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".text"}
_CHUNK_CHARS = 1200


@dataclass
class Chunk:
    chunk_id: str          # e.g. "boiko2023.txt#3"
    source: str
    text: str


@dataclass
class Corpus:
    root: Path
    chunks: list[Chunk] = field(default_factory=list)
    corpus_hash: str = ""

    def chunk_ids(self) -> set[str]:
        return {c.chunk_id for c in self.chunks}

    _retriever = None  # built lazily, cached per corpus

    def keyword_search(self, query: str, k: int = 6, kind: str = "bm25") -> list[Chunk]:
        """Retrieval (long-term-memory read). Defaults to BM25 — term-weighted, length-
        normalised — a real upgrade over raw counts. Pass kind='embedding' for dense retrieval
        when sentence-transformers is installed. The name is kept for backward compatibility."""
        if self._retriever is None:
            from .retrieval import make_retriever
            self._retriever = make_retriever(self.chunks, kind=kind)
        return self._retriever.search(query, k)


def _chunk_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [text[i:i + _CHUNK_CHARS] for i in range(0, len(text), _CHUNK_CHARS)]


def load_corpus(root: str | Path) -> Corpus:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"corpus directory not found: {root}")

    chunks: list[Chunk] = []
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        raw = path.read_bytes()
        hasher.update(path.name.encode("utf-8"))
        hasher.update(raw)
        text = raw.decode("utf-8", errors="replace")
        for i, piece in enumerate(_chunk_text(text)):
            chunks.append(Chunk(chunk_id=f"{path.name}#{i}", source=path.name, text=piece))

    return Corpus(root=root, chunks=chunks, corpus_hash=hasher.hexdigest()[:16])
