"""Automatic large-corpus builder — the SCALE foundation (Kosmos reads ~1,500 papers/run).

Fetches real papers for a research topic from OpenAlex (free, no key, ~250M works) and writes
each as a corpus document (title + abstract + metadata). Paginates with a cursor, so it scales
to thousands of papers, turning the toy local corpus into a Kosmos-magnitude one.

OpenAlex returns abstracts as an inverted index (word -> positions); we reconstruct the text.
Session is injected for offline tests.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .models import Reference

_WORKS = "https://api.openalex.org/works"
_TIMEOUT = 30
_PER_PAGE = 200  # OpenAlex max


def reconstruct_abstract(inv: dict | None) -> str:
    """OpenAlex abstract_inverted_index {word: [positions]} -> plain text."""
    if not inv:
        return ""
    positioned: list[tuple[int, str]] = []
    for word, positions in inv.items():
        for p in positions:
            positioned.append((p, word))
    positioned.sort()
    return " ".join(w for _, w in positioned)


@dataclass
class BuildResult:
    topic: str
    papers: int
    out_dir: str
    references: list[Reference]
    corpus_hash_inputs: int  # number of files written (for a quick sanity check)


class CorpusBuilder:
    def __init__(self, mailto: str | None = None, session: requests.Session | None = None,
                 *, max_retries: int = 2, backoff_base: float = 0.5):
        self.session = session or requests.Session()
        self.mailto = mailto
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def build(self, topic: str, out_dir: str | Path, max_papers: int = 200,
              require_abstract: bool = True, fulltext: bool = False) -> BuildResult:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fetcher = None
        if fulltext:
            from .fulltext import OpenAccessFetcher
            fetcher = OpenAccessFetcher(email=self.mailto)
        refs: list[Reference] = []
        written = 0
        cursor = "*"
        while written < max_papers and cursor:
            page = self._page(topic, cursor)
            if not page:
                break
            cursor = (page.get("meta") or {}).get("next_cursor")
            for work in page.get("results", []):
                if written >= max_papers:
                    break
                abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                if require_abstract and not abstract:
                    continue
                ref = self._to_ref(work)
                full = fetcher.fetch(ref) if fetcher else None  # OA full text where available
                written += self._write(out, work, abstract, full)
                refs.append(ref)
        return BuildResult(topic=topic, papers=written, out_dir=str(out),
                           references=refs, corpus_hash_inputs=written)

    def _page(self, topic: str, cursor: str):
        params = {
            "search": topic,
            "per-page": str(_PER_PAGE),
            "cursor": cursor,
            "sort": "relevance_score:desc",
            "select": "id,doi,title,publication_year,authorships,abstract_inverted_index",
        }
        if self.mailto:
            params["mailto"] = self.mailto
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(_WORKS, params=params, timeout=_TIMEOUT)
            except requests.RequestException:
                resp = None
            if resp is not None and resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return None
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2 ** attempt))
        return None

    @staticmethod
    def _safe_name(work: dict) -> str:
        wid = (work.get("id") or "").rstrip("/").split("/")[-1] or "work"
        return re.sub(r"[^A-Za-z0-9_-]", "", wid) + ".txt"

    @classmethod
    def _write(cls, out: Path, work: dict, abstract: str, full_text: str | None = None) -> int:
        title = work.get("title") or "(untitled)"
        year = work.get("publication_year") or "?"
        authors = ", ".join(
            (a.get("author") or {}).get("display_name", "") for a in work.get("authorships", [])
        )[:400]
        doi = work.get("doi") or ""
        body = (f"Title: {title}\nYear: {year}\nAuthors: {authors}\nDOI: {doi}\n\n"
                f"Abstract: {abstract}\n")
        if full_text:
            body += f"\nFull text (open access):\n{full_text}\n"
        (out / cls._safe_name(work)).write_text(body, encoding="utf-8")
        return 1

    @staticmethod
    def _to_ref(work: dict) -> Reference:
        doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
        authors = [(a.get("author") or {}).get("display_name", "") for a in work.get("authorships", [])]
        return Reference(doi=doi, title=work.get("title"), authors=authors,
                         year=work.get("publication_year"))
