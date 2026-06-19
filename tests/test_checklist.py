"""Offline tests for the 8-gate checklist, using a stub verifier (no network)."""

from __future__ import annotations

from poliresearch.checklist import run_checklist
from poliresearch.citation_verifier import CitationCheck
from poliresearch.models import Claim, ClaimType, Reference


class _StubVerifier:
    """Returns a fixed CitationCheck regardless of input."""

    def __init__(self, ok=True, retracted=False, exists=True):
        self._ok, self._retracted, self._exists = ok, retracted, exists

    def verify(self, ref):
        return CitationCheck(doi=ref.doi or "", exists=self._exists,
                             retracted=self._retracted,
                             authors_match=True if self._ok else False,
                             year_match=True)


def _claim(grounding=("src.txt#0",)):
    return Claim(
        text="ROCK inhibition upregulates ABCA1 in RPE cells.",
        claim_type=ClaimType.SYNTHESIS,
        references=[Reference(doi="10.1/x", authors=["White, A."], year=2025)],
        grounding=list(grounding),
    )


def test_good_claim_blocked_only_by_human_gates():
    verdict = run_checklist(
        _claim(), verifier=_StubVerifier(ok=True),
        corpus_chunk_ids={"src.txt#0"}, corpus_hash="abc123",
        falsification_attempted=True, falsification_survived=True,
        human_reviewed=False,
    )
    # All mechanical gates (1-6) pass; only 7-8 remain (human).
    assert not verdict.failed_gates
    assert {g.gate for g in verdict.pending_human} <= {7, 8}
    assert not verdict.accepted  # human sign-off still required


def test_human_signoff_accepts():
    verdict = run_checklist(
        _claim(), verifier=_StubVerifier(ok=True),
        corpus_chunk_ids={"src.txt#0"}, corpus_hash="abc123",
        falsification_attempted=True, falsification_survived=True,
        human_reviewed=True,
    )
    assert verdict.accepted


def test_retracted_reference_fails():
    verdict = run_checklist(
        _claim(), verifier=_StubVerifier(retracted=True),
        corpus_chunk_ids={"src.txt#0"}, corpus_hash="abc",
        falsification_attempted=True, falsification_survived=True,
        human_reviewed=True,
    )
    assert any(g.gate == 2 and not g.passed for g in verdict.gates)
    assert not verdict.accepted


def test_ungrounded_claim_fails_gate5():
    verdict = run_checklist(
        _claim(grounding=("not_in_corpus#9",)), verifier=_StubVerifier(ok=True),
        corpus_chunk_ids={"src.txt#0"}, corpus_hash="abc",
        falsification_attempted=True, falsification_survived=True, human_reviewed=True,
    )
    assert any(g.gate == 5 and not g.passed for g in verdict.gates)
    assert not verdict.accepted


def test_refuted_claim_never_accepted():
    verdict = run_checklist(
        _claim(), verifier=_StubVerifier(ok=True),
        corpus_chunk_ids={"src.txt#0"}, corpus_hash="abc",
        falsification_attempted=True, falsification_survived=False,  # refuted
        human_reviewed=True,
    )
    assert not verdict.accepted


def test_missing_corpus_hash_fails_gate6():
    verdict = run_checklist(
        _claim(), verifier=_StubVerifier(ok=True),
        corpus_chunk_ids={"src.txt#0"}, corpus_hash=None,
        falsification_attempted=True, falsification_survived=True, human_reviewed=True,
    )
    assert any(g.gate == 6 and not g.passed for g in verdict.gates)
