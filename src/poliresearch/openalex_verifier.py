"""OpenAlex verification — broad coverage + a TITLE-SEARCH fallback.

OpenAlex (free, no key, ~250M works) closes two gaps the Crossref/arXiv anchors leave:
  * it verifies references that have NO DOI and NO arXiv id, by title search (lots of CS
    proceedings, books, theses live only here);
  * it carries an explicit `is_retracted` flag, a second retraction signal beyond Crossref.

Returns the same `CitationCheck` shape as the other verifiers. Session is injected for tests.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import requests

from .citation_verifier import CitationCheck, _titles_match
from .models import Reference

_BY_DOI = "https://api.openalex.org/works/doi:{doi}"
_BY_TITLE = "https://api.openalex.org/works?filter=title.search:{q}&per-page=1"
_TIMEOUT = 15
_RETRY_STATUS = {429, 500, 502, 503, 504}


class OpenAlexVerifier:
    def __init__(self, mailto: str | None = None, session: requests.Session | None = None,
                 *, max_retries: int = 2, backoff_base: float = 0.5):
        self.session = session or requests.Session()
        self.mailto = mailto
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def verify(self, ref: Reference) -> CitationCheck:
        if ref.doi:
            doi = ref.doi.strip().lower()
            for pre in ("https://doi.org/", "http://doi.org/", "doi:"):
                if doi.startswith(pre):
                    doi = doi[len(pre):]
            work = self._get(_BY_DOI.format(doi=doi))
            ident = ref.doi
        elif ref.title:
            res = self._get(_BY_TITLE.format(q=quote(ref.title)))
            arr = res.get("results") or [] if isinstance(res, dict) else []
            work = arr[0] if arr else None      # empty list when nothing matches
            ident = f"title:{ref.title[:40]}"
        else:
            return CitationCheck(doi="", exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="openalex",
                                 error="no DOI or title for OpenAlex lookup")
        if not work:
            return CitationCheck(doi=ident, exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="openalex",
                                 error="not found in OpenAlex (gate 1 fail)")
        return self._evaluate(ident, work, ref)

    def _get(self, url: str):
        if self.mailto:
            url += ("&" if "?" in url else "?") + f"mailto={self.mailto}"
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
            if resp is not None and resp.status_code == 404:
                return None
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2 ** attempt))
        return None

    @staticmethod
    def _evaluate(ident: str, work: dict, ref: Reference) -> CitationCheck:
        title = work.get("title") or work.get("display_name")
        retracted = bool(work.get("is_retracted"))

        authors_match = None
        if ref.author_surnames():
            surnames = set()
            for a in work.get("authorships", []) or []:
                name = (a.get("author") or {}).get("display_name") or ""
                if name:
                    surnames.add(name.strip().split()[-1].lower())
            authors_match = all(s in surnames for s in ref.author_surnames())

        year_match = None
        if ref.year and work.get("publication_year"):
            year_match = abs(int(work["publication_year"]) - int(ref.year)) <= 1

        title_match = None
        if ref.title and title:
            title_match = _titles_match(ref.title, title)

        return CitationCheck(doi=ident, exists=True, retracted=retracted,
                             authors_match=authors_match, year_match=year_match,
                             title_match=title_match, title=title, source="openalex")
