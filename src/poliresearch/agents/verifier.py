"""TieredVerifier — spends verification budget by claim type (TRIZ #3, Local Quality).

The research showed accuracy is wildly uneven: data/literature claims verified well (~85%)
while synthesis claims collapsed (~57.9%). Uniform verification therefore wastes effort on
easy claims and under-checks the dangerous ones. This verifier routes:

  DATA / CITATION  -> mechanical checklist only (Crossref/Retraction Watch + grounding)
  SYNTHESIS        -> mechanical checklist PLUS an extra adversarial falsification pass
"""

from __future__ import annotations

from ..checklist import run_checklist
from ..citation_verifier import CitationVerifier
from ..corpus import Corpus
from ..models import Claim, ClaimType, Verdict
from .falsifier import Falsifier


class TieredVerifier:
    def __init__(self, citation_verifier: CitationVerifier, falsifier: Falsifier):
        self.citations = citation_verifier
        self.falsifier = falsifier

    def verify(self, claim: Claim, corpus: Corpus, *, human_reviewed: bool = False) -> Verdict:
        chunk_ids = corpus.chunk_ids()

        # Synthesis claims get the expensive falsification pass; cheaper claims skip it
        # unless they happen to carry no grounding (then we still probe them).
        do_falsify = claim.claim_type == ClaimType.SYNTHESIS
        attempted = False
        survived: bool | None = None
        if do_falsify:
            ref = self.falsifier.attempt(claim.text, corpus)
            attempted = True
            survived = not ref.refuted
            if not claim.grounding:
                claim.grounding = ref.grounding  # anchor to what the falsifier actually read

        return run_checklist(
            claim,
            verifier=self.citations,
            corpus_chunk_ids=chunk_ids,
            corpus_hash=corpus.corpus_hash,
            falsification_attempted=attempted or not do_falsify,
            falsification_survived=survived if do_falsify else True,
            human_reviewed=human_reviewed,
        )
