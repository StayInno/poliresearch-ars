"""Keyed REST verifiers: NASA ADS, IEEE Xplore, Springer Nature, Elsevier/Scopus.

Each needs an API key and is disabled until its key is set (see sources.py). They share a small
base that does the HTTP + metadata-gate evaluation; subclasses only implement `_fetch` (the
source-specific request + JSON shape -> normalized {title, surnames, year}). None expose a
reliable retraction flag, so gate 2 defers to Crossref/PubMed/OpenAlex. Sessions injected for tests.
"""

from __future__ import annotations

import requests

from .citation_verifier import CitationCheck, _titles_match
from .models import Reference

_TIMEOUT = 20


def _surnames(names: list[str]) -> set[str]:
    out = set()
    for n in names:
        n = (n or "").strip()
        if not n:
            continue
        out.add((n.split(",")[0] if "," in n else n.split()[-1]).strip().lower())
    return out


class _KeyedVerifier:
    source = "keyed"

    def __init__(self, api_key: str | None = None, session: requests.Session | None = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def verify(self, ref: Reference) -> CitationCheck:
        ident = ref.identifier or ""
        rec = self._fetch(ref)
        if not rec:
            return CitationCheck(doi=ident, exists=False, retracted=False, authors_match=None,
                                 year_match=None, source=self.source,
                                 error=f"not found in {self.source}")
        am = None
        if ref.author_surnames():
            am = all(s in rec.get("surnames", set()) for s in ref.author_surnames())
        ym = None
        if ref.year and rec.get("year"):
            ym = abs(int(rec["year"]) - int(ref.year)) <= 1
        tm = _titles_match(ref.title, rec["title"]) if (ref.title and rec.get("title")) else None
        return CitationCheck(doi=ident, exists=True, retracted=False, authors_match=am,
                             year_match=ym, title_match=tm, title=rec.get("title"),
                             source=self.source)

    def _get(self, url, params=None, headers=None):
        try:
            r = self.session.get(url, params=params, headers=headers, timeout=_TIMEOUT)
        except requests.RequestException:
            return None
        if r is None or r.status_code != 200:
            return None
        try:
            return r.json()
        except ValueError:
            return None

    def _fetch(self, ref: Reference) -> dict | None:
        raise NotImplementedError


class ADSVerifier(_KeyedVerifier):
    source = "ads"
    URL = "https://api.adsabs.harvard.edu/v1/search/query"

    def _fetch(self, ref):
        if ref.doi:
            q = f'doi:"{ref.doi}"'
        elif ref.title:
            q = f'title:"{ref.title}"'
        else:
            return None
        data = self._get(self.URL, params={"q": q, "fl": "title,author,year", "rows": 1},
                         headers={"Authorization": f"Bearer {self.api_key}"})
        docs = ((data or {}).get("response", {}) or {}).get("docs", [])
        if not docs:
            return None
        d = docs[0]
        return {"title": (d.get("title") or [None])[0], "surnames": _surnames(d.get("author", [])),
                "year": d.get("year")}


class IEEEVerifier(_KeyedVerifier):
    source = "ieee"
    URL = "http://ieeexploreapi.ieee.org/api/v1/search/articles"

    def _fetch(self, ref):
        params = {"apikey": self.api_key, "max_records": 1, "format": "json"}
        if ref.doi:
            params["doi"] = ref.doi
        elif ref.title:
            params["article_title"] = ref.title
        else:
            return None
        data = self._get(self.URL, params=params)
        arts = (data or {}).get("articles", [])
        if not arts:
            return None
        a = arts[0]
        names = [au.get("full_name", "") for au in ((a.get("authors") or {}).get("authors", []) or [])]
        return {"title": a.get("title"), "surnames": _surnames(names),
                "year": a.get("publication_year")}


class SpringerVerifier(_KeyedVerifier):
    source = "springer"
    URL = "https://api.springernature.com/meta/v2/json"

    def _fetch(self, ref):
        if ref.doi:
            q = f"doi:{ref.doi}"
        elif ref.title:
            q = f'title:"{ref.title}"'
        else:
            return None
        data = self._get(self.URL, params={"q": q, "p": 1, "api_key": self.api_key})
        recs = (data or {}).get("records", [])
        if not recs:
            return None
        r = recs[0]
        names = [c.get("creator", "") for c in (r.get("creators") or [])]
        yr = (r.get("publicationDate") or "")[:4]
        return {"title": r.get("title"), "surnames": _surnames(names),
                "year": int(yr) if yr.isdigit() else None}


class ElsevierVerifier(_KeyedVerifier):
    source = "elsevier"
    URL = "https://api.elsevier.com/content/search/scopus"

    def _fetch(self, ref):
        if ref.doi:
            q = f'DOI("{ref.doi}")'
        elif ref.title:
            q = f'TITLE("{ref.title}")'
        else:
            return None
        data = self._get(self.URL, params={"query": q, "count": 1},
                         headers={"X-ELS-APIKey": self.api_key, "Accept": "application/json"})
        entries = ((data or {}).get("search-results", {}) or {}).get("entry", [])
        if not entries or entries[0].get("error"):
            return None
        e = entries[0]
        names = [e["dc:creator"]] if e.get("dc:creator") else []
        yr = (e.get("prism:coverDate") or "")[:4]
        return {"title": e.get("dc:title"), "surnames": _surnames(names),
                "year": int(yr) if yr.isdigit() else None}
