"""Core data models.

The central TRIZ move (#3 Local Quality) lives here: claims are *typed*, because the
research showed accuracy is wildly uneven by claim type — Kosmos scored ~85% on data
claims but only ~57.9% on synthesis claims. The verifier spends its budget accordingly.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ClaimType(str, enum.Enum):
    """Why a claim type matters: it sets the verification rigor (TRIZ #3)."""

    DATA = "data"          # a number/result computed from the corpus or a dataset
    CITATION = "citation"  # a reference to a specific source (DOI-bearing)
    SYNTHESIS = "synthesis"  # a cross-source inference — the error-prone, high-value kind


@dataclass
class Reference:
    """A literature reference. The truth anchor is a DOI (Crossref) or, for CS/AI papers
    that are arXiv-only, an arXiv id (verified against the arXiv API)."""

    doi: str | None = None
    arxiv_id: str | None = None  # e.g. "2408.06292" — CS/AI papers often have no DOI
    pmid: str | None = None      # PubMed id — biomedical papers
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None

    @property
    def identifier(self) -> str | None:
        if self.doi:
            return self.doi
        if self.arxiv_id:
            return f"arXiv:{self.arxiv_id}"
        if self.pmid:
            return f"PMID:{self.pmid}"
        return f"title:{self.title}" if self.title else None

    def author_surnames(self) -> list[str]:
        out = []
        for a in self.authors:
            a = a.strip()
            if "," in a:           # "Gomes, G."
                out.append(a.split(",")[0].strip().lower())
            elif " " in a:          # "Gabe Gomes"
                out.append(a.split()[-1].strip().lower())
            elif a:
                out.append(a.lower())
        return out


@dataclass
class Claim:
    """One assertion the system makes, with the evidence it rests on."""

    text: str
    claim_type: ClaimType
    references: list[Reference] = field(default_factory=list)
    # IDs of corpus chunks the claim is grounded in (Closed-RAG grounding, gate 4/5).
    grounding: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Claim":
        refs = [Reference(**r) for r in d.get("references", [])]
        return cls(
            text=d["text"],
            claim_type=ClaimType(d.get("claim_type", "synthesis")),
            references=refs,
            grounding=list(d.get("grounding", [])),
        )


@dataclass
class GateResult:
    """Outcome of a single anti-hallucination checklist gate."""

    gate: int
    name: str
    passed: bool
    detail: str = ""
    requires_human: bool = False

    def symbol(self) -> str:
        if self.requires_human and not self.passed:
            return "[HUMAN]"  # awaiting human judgment
        return "[PASS]" if self.passed else "[FAIL]"


@dataclass
class Verdict:
    """Aggregate result for a claim or a whole answer."""

    accepted: bool
    gates: list[GateResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def failed_gates(self) -> list[GateResult]:
        return [g for g in self.gates if not g.passed and not g.requires_human]

    @property
    def pending_human(self) -> list[GateResult]:
        return [g for g in self.gates if g.requires_human and not g.passed]
