"""Open-access full-text fetcher — deeper grounding, legally.

Pulls full text ONLY where it is openly available, falling back to the abstract otherwise:
  * arXiv preprints  -> ar5iv HTML (full text, no key)
  * DOIs             -> Unpaywall finds the legal OA copy (HTML stripped, or PDF if `pypdf` present)

Paywalled papers are never scraped — that stays metadata + abstract. Session injected for tests.
"""

from __future__ import annotations

import re

import requests

from .models import Reference

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TIMEOUT = 30


def strip_html(html: str) -> str:
    html = _SCRIPT.sub(" ", html)
    return _WS.sub(" ", _TAG.sub(" ", html)).strip()


class OpenAccessFetcher:
    def __init__(self, email: str | None = None, session: requests.Session | None = None,
                 *, max_chars: int = 20000, core=None):
        self.session = session or requests.Session()
        self.email = email or "poliresearch@example.com"
        self.max_chars = max_chars
        # arXiv/ar5iv block the default python-requests UA; identify politely.
        if hasattr(self.session, "headers"):
            self.session.headers.setdefault(
                "User-Agent", f"PoliResearch/0.1 (mailto:{self.email})")
        # Optional CORE full-text source — only if CORE_API_KEY is set (disabled otherwise).
        self.core = core
        if self.core is None:
            import os
            k = os.environ.get("CORE_API_KEY")
            if k:
                from .core_verifier import CoreVerifier
                self.core = CoreVerifier(api_key=k)

    def fetch(self, ref: Reference) -> str | None:
        if ref.arxiv_id:
            t = self._arxiv(ref.arxiv_id)
            if t:
                return t
        if ref.doi:
            t = self._unpaywall(ref.doi)
            if t:
                return t
        if self.core:  # CORE OA full text (keyed) as a final fallback
            try:
                t = self.core.fetch_fulltext(ref)
                if t:
                    return t[:self.max_chars]
            except Exception:
                pass
        return None

    def _get(self, url: str):
        try:
            r = self.session.get(url, timeout=_TIMEOUT)
        except requests.RequestException:
            return None
        return r if (r is not None and r.status_code == 200) else None

    _ARXIV_ERR = ("no content available", "conversion to html had a fatal error",
                  "document may be truncated or damaged")

    def _arxiv(self, aid: str) -> str | None:
        aid = aid.strip()
        for pre in ("arxiv:", "arXiv:"):
            if aid.lower().startswith(pre):
                aid = aid[len(pre):]
        # Native arXiv HTML (2023+) first, then ar5iv as fallback; reject conversion-error pages.
        for url in (f"https://arxiv.org/html/{aid}", f"https://ar5iv.org/abs/{aid}"):
            r = self._get(url)
            if not r:
                continue
            text = strip_html(r.text)
            low = text.lower()
            if len(text) > 500 and not any(e in low for e in self._ARXIV_ERR):
                return text[:self.max_chars]
        return None

    def _unpaywall(self, doi: str) -> str | None:
        doi = doi.replace("https://doi.org/", "").strip()
        meta = self._get(f"https://api.unpaywall.org/v2/{doi}?email={self.email}")
        if not meta:
            return None
        try:
            loc = (meta.json() or {}).get("best_oa_location") or {}
        except ValueError:
            return None
        url = loc.get("url_for_pdf") or loc.get("url")
        if not url:
            return None
        if url.lower().endswith(".pdf"):
            return self._pdf(url)
        r = self._get(url)
        if not r:
            return None
        text = strip_html(r.text)
        return text[:self.max_chars] if len(text) > 200 else None

    def _pdf(self, url: str) -> str | None:
        try:
            import io
            import pypdf  # optional dependency
        except Exception:
            return None  # no PDF backend -> caller falls back to abstract
        r = self._get(url)
        if not r:
            return None
        try:
            reader = pypdf.PdfReader(io.BytesIO(r.content))
            text = " ".join((p.extract_text() or "") for p in reader.pages)
            return text[:self.max_chars] if len(text) > 200 else None
        except Exception:
            return None
