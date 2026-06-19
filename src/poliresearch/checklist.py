"""The Anti-Hallucination Checklist — deck slide 14, the system's acceptance gate.

Eight gates. The TRIZ analysis reassigned the labor: gates 1-6 are fully mechanical,
gate 7 (falsification) is *automated* by the Falsifier agent (TRIZ #13 inversion) with
human confirmation, and gate 8 (domain significance) stays human. A claim is ACCEPTED
only when no mechanical gate fails; gates 7-8 surface as pending-human, never auto-passed.

    1  DOI resolves via Crossref                  [auto]
    2  Not retracted (Retraction Watch)           [auto]
    3  Authors + year match the original          [auto]
    4  Every claim has an inline citation          [auto]
    5  Grounded in corpus, not parametric memory   [auto]  (Closed RAG)
    6  Pipeline reproducible (corpus hash present) [auto]  (DVC/MLflow)
    7  Counter-evidence searched (falsification)   [auto+human]  (FVA-RAG)
    8  Domain expert reviewed                       [human, mandatory]
"""

from __future__ import annotations

from .citation_verifier import CitationVerifier
from .models import Claim, ClaimType, GateResult, Verdict


def run_checklist(
    claim: Claim,
    *,
    verifier: CitationVerifier | None = None,
    corpus_chunk_ids: set[str] | None = None,
    corpus_hash: str | None = None,
    falsification_attempted: bool = False,
    falsification_survived: bool | None = None,
    human_reviewed: bool = False,
) -> Verdict:
    """Evaluate one claim against all eight gates and return a Verdict."""
    verifier = verifier or CitationVerifier()
    corpus_chunk_ids = corpus_chunk_ids or set()
    gates: list[GateResult] = []
    notes: list[str] = []

    # Citation gates (1-3) apply only to claims that carry references.
    needs_citation = claim.claim_type in (ClaimType.CITATION, ClaimType.SYNTHESIS, ClaimType.DATA)

    if claim.references:
        all_exist = all_not_retracted = all_meta_ok = True
        for ref in claim.references:
            chk = verifier.verify(ref)
            if not chk.exists:
                all_exist = False
                notes.append(f"DOI {ref.doi!r}: {chk.error or 'not found'}")
            if chk.retracted:
                all_not_retracted = False
                notes.append(f"DOI {chk.doi}: RETRACTED")
            if chk.authors_match is False or chk.year_match is False:
                all_meta_ok = False
                notes.append(f"DOI {chk.doi}: author/year mismatch vs Crossref")
            if chk.title_match is False:
                all_meta_ok = False
                notes.append(f"DOI {chk.doi}: title mismatch vs Crossref (wrong-DOI risk)")
        gates.append(GateResult(1, "Identifier resolves (Crossref/arXiv)", all_exist))
        gates.append(GateResult(2, "Not retracted", all_not_retracted))
        gates.append(GateResult(3, "Authors + year + title match", all_meta_ok))
    else:
        # No references attached. For citation/synthesis claims that is itself a gate-4 failure.
        gates.append(GateResult(1, "Identifier resolves (Crossref/arXiv)", not needs_citation,
                                detail="no references attached"))
        gates.append(GateResult(2, "Not retracted", True))
        gates.append(GateResult(3, "Authors + year match", True))

    # Gate 4 — every claim must carry at least one inline citation.
    has_citation = bool(claim.references)
    gates.append(GateResult(4, "Inline citation present", has_citation,
                            detail="" if has_citation else "claim has no citation"))

    # Gate 5 — grounded in the closed corpus, not parametric memory.
    grounded = bool(claim.grounding) and all(g in corpus_chunk_ids for g in claim.grounding)
    detail5 = ""
    if not claim.grounding:
        detail5 = "no corpus grounding ids — possible parametric-memory answer"
    elif not grounded:
        bad = [g for g in claim.grounding if g not in corpus_chunk_ids]
        detail5 = f"grounding ids not in corpus: {bad}"
    gates.append(GateResult(5, "Grounded in corpus (Closed RAG)", grounded, detail=detail5))

    # Gate 6 — reproducibility: a corpus hash must be recorded for this run.
    repro = bool(corpus_hash)
    gates.append(GateResult(6, "Reproducible (corpus hash)", repro,
                            detail="" if repro else "no corpus hash recorded"))

    # Gate 7 — counter-evidence searched (automated by the Falsifier; human confirms).
    g7_pass = falsification_attempted and (falsification_survived is True) and human_reviewed
    g7_detail = (
        "not yet run" if not falsification_attempted
        else "claim was REFUTED by falsifier" if falsification_survived is False
        else "survived falsification — awaiting human sign-off"
        if not human_reviewed else "confirmed"
    )
    gates.append(GateResult(7, "Counter-evidence searched (FVA-RAG)", g7_pass,
                            detail=g7_detail, requires_human=not g7_pass and falsification_survived is not False))

    # Gate 8 — domain expert review. Mandatory human step; never auto-passes.
    gates.append(GateResult(8, "Domain expert reviewed", human_reviewed,
                            detail="" if human_reviewed else "mandatory human sign-off pending",
                            requires_human=not human_reviewed))

    mechanical_fail = any(
        not g.passed and not g.requires_human and g.gate in (1, 2, 3, 4, 5, 6)
        for g in gates
    )
    # A claim is accepted only if no mechanical gate failed AND it was not refuted.
    refuted = falsification_survived is False
    accepted = (not mechanical_fail) and (not refuted) and human_reviewed
    return Verdict(accepted=accepted, gates=gates, notes=notes)


def format_verdict(claim: Claim, verdict: Verdict) -> str:
    lines = [f"Claim ({claim.claim_type.value}): {claim.text}"]
    for g in verdict.gates:
        suffix = f"  - {g.detail}" if g.detail else ""
        lines.append(f"  {g.symbol():8} Gate {g.gate}: {g.name}{suffix}")
    status = "ACCEPTED" if verdict.accepted else "NOT ACCEPTED"
    lines.append(f"  => {status}")
    if verdict.notes:
        lines.append("  notes: " + "; ".join(verdict.notes))
    return "\n".join(lines)
