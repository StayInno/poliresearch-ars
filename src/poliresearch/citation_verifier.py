"""Citation verification against external truth anchors.

The load-bearing, no-LLM-needed part of the system (deck slides 12 & 14, TRIZ #24
Intermediary + #28 external truth). Implements checklist gates 1-3:

  Gate 1  — does the DOI resolve to a real work?              (Crossref)
  Gate 2  — has the work been retracted?                      (Crossref / Retraction Watch)
  Gate 3a — do the claimed authors + year match?              (Crossref metadata)
  Gate 3b — does the claimed TITLE match the DOI's real title? (catches wrong-DOI / hijacking)

Retraction Watch was absorbed by Crossref in 2023, so retraction status surfaces inside the
standard work record via `type`, `update-to` relations of type "retraction", `update-policy`
labels, and the `relation` map ("is-retracted-by"). We check all four — broader coverage than
relying on `update-to` alone.

Robustness: transient Crossref errors (429/5xx) are retried with exponential backoff; results
are cached per-DOI; `verify_many` runs concurrently. Network access is injected (`session`) so
tests run fully offline.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import requests

from .models import Reference

CROSSREF_WORKS = "https://api.crossref.org/works/{doi}"
_TIMEOUT = 15
_RETRY_STATUS = {429, 500, 502, 503, 504}
# Title is a *corroborating* signal: its job is to catch gross mismatches (a real DOI attached
# to a totally unrelated paper), not to adjudicate paraphrases. We therefore use an order-
# independent token-overlap coefficient (robust to reordering/abbreviation) with a modest
# threshold, rather than char-level similarity which punished faithful paraphrases.
_TITLE_THRESHOLD = 0.45
_TITLE_STOPWORDS = {
    "a", "an", "the", "of", "for", "with", "and", "to", "in", "on", "at", "by", "from",
    "using", "use", "via", "towards", "toward", "into", "over", "as",
}


@dataclass
class CitationCheck:
    doi: str               # the identifier checked (a DOI, or "arXiv:..." for arXiv refs)
    exists: bool
    retracted: bool
    authors_match: bool | None  # None when we had nothing to compare against
    year_match: bool | None
    title_match: bool | None = None
    title: str | None = None
    error: str | None = None
    source: str = "crossref"  # which truth anchor answered: "crossref" | "arxiv"

    @property
    def ok(self) -> bool:
        """Trustworthy iff it exists, is not retracted, and every metadata field we
        *could* check did not contradict the claim."""
        if not self.exists or self.retracted:
            return False
        if self.authors_match is False or self.year_match is False:
            return False
        if self.title_match is False:
            return False
        return True


def _title_tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower())
            if len(t) > 1 and t not in _TITLE_STOPWORDS}


def _title_overlap(a: str, b: str) -> float:
    """Overlap coefficient = |A ∩ B| / min(|A|, |B|) over content tokens. Order-independent
    and tolerant of abbreviation/extra words, while a totally unrelated title scores ~0."""
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _titles_match(claimed: str, actual: str) -> bool:
    return _title_overlap(claimed, actual) >= _TITLE_THRESHOLD


class CitationVerifier:
    def __init__(
        self,
        mailto: str | None = None,
        session: requests.Session | None = None,
        *,
        max_retries: int = 3,
        backoff_base: float = 0.5,
        max_workers: int = 8,
    ):
        self.session = session or requests.Session()
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_workers = max_workers
        # Cache the network result per DOI: a dict = the fetched Crossref record;
        # a CitationCheck = a terminal error (404 / transient failure). Per-ref metadata
        # gates are always recomputed, so two refs sharing a DOI are evaluated correctly.
        self._cache: dict[str, dict | CitationCheck] = {}
        # Crossref "polite pool": identifying yourself yields faster, more reliable service.
        ua = "PoliResearch/0.1 (https://github.com/ai-poliresearch)"
        if mailto:
            ua += f" mailto:{mailto}"
        self.session.headers.setdefault("User-Agent", ua)

    # --- public API ---
    def verify(self, ref: Reference) -> CitationCheck:
        """Run gates 1-3 for one reference (cached per DOI)."""
        if not ref.doi:
            return CitationCheck(
                doi="", exists=False, retracted=False, authors_match=None, year_match=None,
                error="no DOI supplied - a citation without a verifiable DOI cannot pass gate 1",
            )
        doi = self._normalise(ref.doi)
        if doi not in self._cache:
            self._cache[doi] = self._fetch(doi)
        record = self._cache[doi]
        if isinstance(record, CitationCheck):   # terminal error (404 / transient)
            return record
        return self._evaluate(doi, record, ref)  # always per-ref

    def verify_many(self, refs: list[Reference]) -> list[CitationCheck]:
        """Concurrent verification — a long bibliography no longer blocks serially."""
        if len(refs) <= 1:
            return [self.verify(r) for r in refs]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.verify, refs))

    # --- internals ---
    @staticmethod
    def _normalise(doi: str) -> str:
        doi = doi.strip().lower()
        for pre in ("https://doi.org/", "http://doi.org/", "doi:"):
            if doi.startswith(pre):
                doi = doi[len(pre):]
        return doi

    def _get_with_retry(self, url: str) -> requests.Response | Exception:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=_TIMEOUT)
            except requests.RequestException as e:
                last_exc = e
            else:
                if resp.status_code not in _RETRY_STATUS:
                    return resp
                last_exc = RuntimeError(f"HTTP {resp.status_code}")
            if attempt < self.max_retries:
                time.sleep(self.backoff_base * (2 ** attempt))
        return last_exc or RuntimeError("request failed")

    def _fetch(self, doi: str) -> dict | CitationCheck:
        """Return the Crossref record (dict) or a terminal-error CitationCheck."""
        result = self._get_with_retry(CROSSREF_WORKS.format(doi=doi))
        if isinstance(result, Exception):
            return CitationCheck(doi=doi, exists=False, retracted=False,
                                 authors_match=None, year_match=None,
                                 error=f"network/transient error after retries: {result}")
        resp = result
        if resp.status_code == 404:
            return CitationCheck(doi=doi, exists=False, retracted=False,
                                 authors_match=None, year_match=None,
                                 error="DOI not found in Crossref (gate 1 fail)")
        if resp.status_code != 200:
            return CitationCheck(doi=doi, exists=False, retracted=False,
                                 authors_match=None, year_match=None,
                                 error=f"Crossref returned HTTP {resp.status_code}")
        return resp.json().get("message", {})

    @staticmethod
    def _detect_retraction(msg: dict) -> bool:
        if str(msg.get("type", "")).lower() in {"retraction", "retracted"}:
            return True
        for upd in msg.get("update-to", []) or []:
            if "retract" in str(upd.get("type", "")).lower():
                return True
        for label in msg.get("update-policy", []) or []:
            if "retract" in str(label).lower():
                return True
        # `relation` map carries RW links such as "is-retracted-by" on the original work.
        for rel_key in (msg.get("relation", {}) or {}):
            if "retract" in str(rel_key).lower():
                return True
        return False

    @classmethod
    def _evaluate(cls, doi: str, msg: dict, ref: Reference) -> CitationCheck:
        cr_title = (msg.get("title") or [None])[0]
        retracted = cls._detect_retraction(msg)

        # Gate 3a: authors
        authors_match: bool | None = None
        if ref.author_surnames():
            cr_surnames = {
                (a.get("family") or "").strip().lower()
                for a in msg.get("author", []) or [] if a.get("family")
            }
            # EVERY claimed surname must be a real author. `any()` let a citation with one real
            # + one fabricated co-author pass (a measured false positive); `all()` closes that.
            authors_match = all(s in cr_surnames for s in ref.author_surnames())

        # Gate 3a: year
        year_match: bool | None = None
        if ref.year:
            parts = (msg.get("published-print") or msg.get("published-online")
                     or msg.get("issued") or {}).get("date-parts", [[None]])
            cr_year = parts[0][0] if parts and parts[0] else None
            if cr_year:
                year_match = abs(int(cr_year) - int(ref.year)) <= 1  # online/print slip

        # Gate 3b: title — catches a real DOI attached to the WRONG paper, while tolerating
        # faithful paraphrases (order-independent token overlap).
        title_match: bool | None = None
        if ref.title and cr_title:
            title_match = _titles_match(ref.title, cr_title)

        return CitationCheck(doi=doi, exists=True, retracted=retracted,
                             authors_match=authors_match, year_match=year_match,
                             title_match=title_match, title=cr_title)
