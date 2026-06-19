"""Semantic Scholar verifier — keyed source (disabled unless SEMANTIC_SCHOLAR_API_KEY is set).

Verifies by DOI, arXiv id, or title search. S2 has no retraction flag, so gate 2 is reported
not-retracted (rely on Crossref/PubMed/OpenAlex for retraction). Session injected for tests.
"""

from __future__ import annotations

from urllib.parse import quote

import requests

from .citation_verifier import CitationCheck, _titles_match
from .models import Reference

_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_FIELDS = "title,year,authors"
_TIMEOUT = 15


class SemanticScholarVerifier:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def _headers(self):
        return {"x-api-key": self.api_key} if self.api_key else {}

    def verify(self, ref: Reference) -> CitationCheck:
        if ref.doi:
            url, ident = f"{_BASE}/DOI:{ref.doi}?fields={_FIELDS}", ref.doi
        elif ref.arxiv_id:
            url, ident = f"{_BASE}/arXiv:{ref.arxiv_id}?fields={_FIELDS}", f"arXiv:{ref.arxiv_id}"
        elif ref.title:
            url = f"{_BASE}/search?query={quote(ref.title)}&limit=1&fields={_FIELDS}"
            ident = f"title:{ref.title[:40]}"
        else:
            return CitationCheck(doi="", exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="semanticscholar",
                                 error="no DOI/arXiv/title for Semantic Scholar")
        data = self._get(url)
        if not isinstance(data, dict):
            paper = None
        elif "search?" in url:
            arr = data.get("data") or []        # empty list when nothing matches
            paper = arr[0] if arr else None
        else:
            paper = data if data.get("title") else None
        if not paper:
            return CitationCheck(doi=ident, exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="semanticscholar",
                                 error="not found in Semantic Scholar")
        return self._evaluate(ident, paper, ref)

    def _get(self, url: str):
        try:
            r = self.session.get(url, headers=self._headers(), timeout=_TIMEOUT)
        except requests.RequestException:
            return None
        if r is not None and r.status_code == 200:
            try:
                return r.json()
            except ValueError:
                return None
        return None

    @staticmethod
    def _evaluate(ident: str, paper: dict, ref: Reference) -> CitationCheck:
        title = paper.get("title")
        authors_match = None
        if ref.author_surnames():
            surnames = {(a.get("name") or "").strip().split()[-1].lower()
                        for a in paper.get("authors", []) or [] if a.get("name")}
            authors_match = all(s in surnames for s in ref.author_surnames())
        year_match = None
        if ref.year and paper.get("year"):
            year_match = abs(int(paper["year"]) - int(ref.year)) <= 1
        title_match = _titles_match(ref.title, title) if (ref.title and title) else None
        return CitationCheck(doi=ident, exists=True, retracted=False, authors_match=authors_match,
                             year_match=year_match, title_match=title_match, title=title,
                             source="semanticscholar")
