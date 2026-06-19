"""Unified reference verifier — routes each reference to the right truth anchor.

This is what makes the verification layer domain-universal. Multiple paper sources, one shape:

  DOI            -> Crossref   (richest metadata + Retraction Watch); OpenAlex fallback on 404
  arXiv id       -> arXiv API  (CS/AI preprints with no DOI)
  PMID           -> PubMed     (biomedical papers)
  title only     -> OpenAlex   (proceedings/books/theses with no identifier at all)

The checklist and pipeline depend only on `.verify(ref)`, so adding another anchor (DBLP, ACL
Anthology, Semantic Scholar) is one more router branch — nothing downstream changes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from .arxiv_verifier import ArxivVerifier
from .citation_verifier import CitationCheck, CitationVerifier
from .models import Reference
from .openalex_verifier import OpenAlexVerifier
from .pubmed_verifier import PubMedVerifier


class UnifiedVerifier:
    def __init__(self, mailto: str | None = None, *, max_workers: int = 8):
        self.crossref = CitationVerifier(mailto=mailto, max_workers=max_workers)
        self.arxiv = ArxivVerifier()
        self.pubmed = PubMedVerifier()
        self.openalex = OpenAlexVerifier(mailto=mailto)
        self.max_workers = max_workers
        # Keyed sources are DISABLED unless their API key is set (see sources.py).
        from .sources import enabled_sources
        self.keyed = [s.build() for s in enabled_sources()]

    def _keyed_fallback(self, ref: Reference) -> CitationCheck | None:
        for v in self.keyed:
            try:
                chk = v.verify(ref)
            except Exception:
                continue
            if chk.exists:
                return chk
        return None

    def verify(self, ref: Reference) -> CitationCheck:
        if ref.doi:
            chk = self.crossref.verify(ref)
            # Cross-source fallback: Crossref -> OpenAlex -> any enabled keyed source.
            if not chk.exists:
                alt = self.openalex.verify(ref)
                if alt.exists:
                    return alt
                keyed = self._keyed_fallback(ref)
                if keyed:
                    return keyed
            return chk
        if ref.arxiv_id:
            chk = self.arxiv.verify(ref)
            return chk if chk.exists else (self._keyed_fallback(ref) or chk)
        if ref.pmid:
            return self.pubmed.verify(ref)
        if ref.title:  # no identifier at all -> OpenAlex title search, then keyed sources
            chk = self.openalex.verify(ref)
            return chk if chk.exists else (self._keyed_fallback(ref) or chk)
        return CitationCheck(doi="", exists=False, retracted=False, authors_match=None,
                             year_match=None,
                             error="no DOI / arXiv id / PMID / title - cannot verify (gate 1)")

    def verify_many(self, refs: list[Reference]) -> list[CitationCheck]:
        if len(refs) <= 1:
            return [self.verify(r) for r in refs]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(self.verify, refs))
