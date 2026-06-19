"""CORE verifier — keyed source (disabled unless CORE_API_KEY is set).

CORE (core.ac.uk) is the largest aggregator of open-access full text. Verifies by DOI or title;
also exposes OA full text. Requires a free API key. Session injected for tests.
"""

from __future__ import annotations

import requests

from .citation_verifier import CitationCheck, _titles_match
from .models import Reference

_SEARCH = "https://api.core.ac.uk/v3/search/works"
_TIMEOUT = 20


class CoreVerifier:
    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def verify(self, ref: Reference) -> CitationCheck:
        if ref.doi:
            q, ident = f'doi:"{ref.doi}"', ref.doi
        elif ref.title:
            q, ident = f'title:"{ref.title}"', f"title:{ref.title[:40]}"
        else:
            return CitationCheck(doi="", exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="core", error="no DOI/title for CORE")
        work = self._search(q)
        if not work:
            return CitationCheck(doi=ident, exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="core", error="not found in CORE")
        return self._evaluate(ident, work, ref)

    def fetch_fulltext(self, ref: Reference) -> str | None:
        q = f'doi:"{ref.doi}"' if ref.doi else (f'title:"{ref.title}"' if ref.title else None)
        if not q:
            return None
        work = self._search(q)
        text = (work or {}).get("fullText")
        return text if text and len(text) > 200 else None

    def _search(self, q: str):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            r = self.session.get(_SEARCH, params={"q": q, "limit": 1}, headers=headers,
                                 timeout=_TIMEOUT)
        except requests.RequestException:
            return None
        if r is None or r.status_code != 200:
            return None
        try:
            results = (r.json() or {}).get("results", [])
        except ValueError:
            return None
        return results[0] if results else None

    @staticmethod
    def _evaluate(ident: str, work: dict, ref: Reference) -> CitationCheck:
        title = work.get("title")
        authors_match = None
        if ref.author_surnames():
            surnames = set()
            for a in work.get("authors", []) or []:
                name = a.get("name") if isinstance(a, dict) else str(a)
                if name:
                    last = name.split(",")[0] if "," in name else name.split()[-1]
                    surnames.add(last.strip().lower())
            authors_match = all(s in surnames for s in ref.author_surnames())
        year_match = None
        if ref.year and work.get("yearPublished"):
            year_match = abs(int(work["yearPublished"]) - int(ref.year)) <= 1
        title_match = _titles_match(ref.title, title) if (ref.title and title) else None
        return CitationCheck(doi=ident, exists=True, retracted=False, authors_match=authors_match,
                             year_match=year_match, title_match=title_match, title=title,
                             source="core")
