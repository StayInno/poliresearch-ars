"""arXiv verification — the CS/AI truth anchor.

Computer-science and AI papers usually live on arXiv and frequently have *no DOI*, so the
Crossref path cannot verify them. This verifier resolves an arXiv id against the free arXiv
API (Atom XML, no key) and returns the same `CitationCheck` shape as the Crossref verifier, so
the checklist and pipeline treat both identically.

Gates covered: 1 (id resolves), 3a (authors + year), 3b (title match). arXiv has no retraction
register, so gate 2 is reported as not-retracted with a documented caveat — withdrawn papers are
detected when arXiv returns no usable entry.
"""

from __future__ import annotations

import time
from xml.etree import ElementTree as ET

import requests

from .citation_verifier import CitationCheck, _titles_match
from .models import Reference

ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}"
_ATOM = "{http://www.w3.org/2005/Atom}"
_TIMEOUT = 15
_RETRY_STATUS = {429, 500, 502, 503, 504}


class ArxivVerifier:
    def __init__(self, session: requests.Session | None = None, *,
                 max_retries: int = 3, backoff_base: float = 0.5):
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._cache: dict[str, ET.Element | None] = {}

    @staticmethod
    def _normalise(arxiv_id: str) -> str:
        aid = arxiv_id.strip()
        for pre in ("arxiv:", "arXiv:", "https://arxiv.org/abs/", "http://arxiv.org/abs/"):
            if aid.lower().startswith(pre.lower()):
                aid = aid[len(pre):]
        return aid.rstrip("/")

    def verify(self, ref: Reference) -> CitationCheck:
        if not ref.arxiv_id:
            return CitationCheck(doi="", exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="arxiv",
                                 error="no arXiv id supplied")
        aid = self._normalise(ref.arxiv_id)
        if aid not in self._cache:
            self._cache[aid] = self._fetch_entry(aid)
        entry = self._cache[aid]
        ident = f"arXiv:{aid}"
        if entry is None:
            return CitationCheck(doi=ident, exists=False, retracted=False, authors_match=None,
                                 year_match=None, source="arxiv",
                                 error="arXiv id not found / withdrawn (gate 1 fail)")
        return self._evaluate(ident, entry, ref)

    def _fetch_entry(self, aid: str) -> ET.Element | None:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(ARXIV_API.format(id=aid), timeout=_TIMEOUT)
            except requests.RequestException:
                resp = None
            if resp is not None and resp.status_code == 200:
                return self._parse_feed(resp.text)
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2 ** attempt))
        return None

    @staticmethod
    def _parse_feed(xml_text: str) -> ET.Element | None:
        try:
            feed = ET.fromstring(xml_text)
        except ET.ParseError:
            return None
        entry = feed.find(f"{_ATOM}entry")
        if entry is None:
            return None
        title_el = entry.find(f"{_ATOM}title")
        title = (title_el.text or "").strip() if title_el is not None else ""
        # arXiv returns a pseudo-entry titled "Error" for unknown ids.
        if title.lower() == "error" or not title:
            return None
        return entry

    @classmethod
    def _evaluate(cls, ident: str, entry: ET.Element, ref: Reference) -> CitationCheck:
        title_el = entry.find(f"{_ATOM}title")
        ar_title = (title_el.text or "").strip() if title_el is not None else None

        # authors
        authors_match: bool | None = None
        if ref.author_surnames():
            ar_surnames = set()
            for au in entry.findall(f"{_ATOM}author"):
                name_el = au.find(f"{_ATOM}name")
                if name_el is not None and name_el.text:
                    ar_surnames.add(name_el.text.strip().split()[-1].lower())
            authors_match = all(s in ar_surnames for s in ref.author_surnames())

        # year (from <published>YYYY-...>)
        year_match: bool | None = None
        if ref.year:
            pub_el = entry.find(f"{_ATOM}published")
            if pub_el is not None and pub_el.text:
                ar_year = int(pub_el.text[:4])
                year_match = abs(ar_year - int(ref.year)) <= 1

        # title
        title_match: bool | None = None
        if ref.title and ar_title:
            title_match = _titles_match(ref.title, ar_title)

        return CitationCheck(doi=ident, exists=True, retracted=False,
                             authors_match=authors_match, year_match=year_match,
                             title_match=title_match, title=ar_title, source="arxiv")
