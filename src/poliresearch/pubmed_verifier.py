"""PubMed verification — the biomedical truth anchor (PMID).

Uses NCBI E-utilities `esummary` (free, no key). Returns the same `CitationCheck` shape. PubMed
flags retracted articles with the publication type "Retracted Publication", which we check for
gate 2. Session is injected for tests.
"""

from __future__ import annotations

import time

import requests

from .citation_verifier import _titles_match
from .citation_verifier import CitationCheck
from .models import Reference

_ESUMMARY = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
             "?db=pubmed&retmode=json&id={pmid}")
_TIMEOUT = 15


class PubMedVerifier:
    def __init__(self, session: requests.Session | None = None, *,
                 max_retries: int = 2, backoff_base: float = 0.5):
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def verify(self, ref: Reference) -> CitationCheck:
        if not ref.pmid:
            return CitationCheck(doi="", exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="pubmed", error="no PMID supplied")
        pmid = str(ref.pmid).strip().lower().replace("pmid:", "").strip()
        data = self._get(_ESUMMARY.format(pmid=pmid))
        ident = f"PMID:{pmid}"
        rec = ((data or {}).get("result", {}) or {}).get(pmid) if isinstance(data, dict) else None
        if not rec or rec.get("error"):
            return CitationCheck(doi=ident, exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="pubmed",
                                 error="PMID not found in PubMed (gate 1 fail)")
        return self._evaluate(ident, rec, ref)

    def _get(self, url: str):
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=_TIMEOUT)
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
    def _evaluate(ident: str, rec: dict, ref: Reference) -> CitationCheck:
        title = rec.get("title")
        pubtypes = [str(t).lower() for t in (rec.get("pubtype") or [])]
        retracted = any("retract" in t for t in pubtypes)

        authors_match = None
        if ref.author_surnames():
            surnames = {(a.get("name") or "").strip().split()[0].lower()  # "Boiko DA" -> boiko
                        for a in rec.get("authors", []) or [] if a.get("name")}
            authors_match = all(s in surnames for s in ref.author_surnames())

        year_match = None
        pubdate = rec.get("pubdate") or rec.get("sortpubdate") or ""
        if ref.year and pubdate[:4].isdigit():
            year_match = abs(int(pubdate[:4]) - int(ref.year)) <= 1

        title_match = None
        if ref.title and title:
            title_match = _titles_match(ref.title, title)

        return CitationCheck(doi=ident, exists=True, retracted=retracted,
                             authors_match=authors_match, year_match=year_match,
                             title_match=title_match, title=title, source="pubmed")
