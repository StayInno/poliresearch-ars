"""Domain profiles — what makes the system *universal* with CS/AI shipped *first*.

The architecture (open generator -> closed multi-vote falsifier -> verification gates -> human)
is domain-independent. What changes between fields is captured here in a `Domain` profile:

  * which truth anchors verify a citation (CS/AI: arXiv + DOI; biomed: DOI + PubMed + RetractionWatch)
  * which sources to search and reason over
  * whether hypotheses can be tested *empirically by running code* — true for CS/AI, false for
    fields whose experiments are physical (the single biggest reason CS/AI can reach a real
    closed loop while biomedicine stops at in-vitro)
  * domain framing injected into the generator/falsifier prompts

Ship a new field by adding a profile here. Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Domain:
    key: str
    name: str
    description: str
    source_hints: tuple[str, ...]        # named in generator/search prompts
    id_anchors: tuple[str, ...]          # verification anchors, in preference order
    enable_experiments: bool             # can hypotheses be tested by executing code?
    experiment_language: str | None = field(default=None)
    prompt_framing: str = ""             # domain context injected into agent system prompts


# --- First-class profile: Computer Science & AI ---
CS_AI = Domain(
    key="cs_ai",
    name="Computer Science & AI",
    description=(
        "Autonomous research over the CS/AI literature with empirical, code-based verification. "
        "Hypotheses about algorithms, models, and systems are not only checked against the "
        "literature but TESTED by writing and running code."
    ),
    source_hints=("arXiv (cs.AI, cs.LG, cs.CL, cs.CV)", "OpenReview", "ACL Anthology",
                  "Papers with Code", "Semantic Scholar", "DBLP"),
    id_anchors=("arxiv", "doi", "openalex"),  # arXiv first; OpenAlex title-search for no-id papers
    enable_experiments=True,
    experiment_language="python",
    prompt_framing=(
        "Domain: Computer Science & AI. Most sources are arXiv preprints (cite by arXiv id) and "
        "peer-reviewed proceedings (NeurIPS, ICML, ICLR, ACL, CVPR). Where a hypothesis is "
        "computational, prefer claims that can be settled by running a small, self-contained "
        "experiment over claims that can only be argued from text."
    ),
)

# --- Universality proof: a generic profile with no code execution ---
GENERIC = Domain(
    key="generic",
    name="Generic / literature-only",
    description="Closed-corpus literature synthesis for any field; no code experiments.",
    source_hints=("Crossref-indexed journals", "Semantic Scholar"),
    id_anchors=("doi", "arxiv"),
    enable_experiments=False,
    prompt_framing="Domain: general scientific literature. Cite by DOI where possible.",
)

# --- Stub showing how a physical-science field plugs in ---
BIOMED = Domain(
    key="biomed",
    name="Biomedicine",
    description="Biomedical literature; experiments are physical (wet-lab), so no code loop.",
    source_hints=("PubMed", "Crossref", "bioRxiv", "Retraction Watch"),
    id_anchors=("doi", "pubmed", "openalex"),
    enable_experiments=False,
    prompt_framing=(
        "Domain: biomedicine. Cite by DOI. Hypotheses require physical validation (in vitro / "
        "in vivo); the system cannot run the experiment, so flag every claim as needing wet-lab "
        "confirmation."
    ),
)

_REGISTRY = {d.key: d for d in (CS_AI, GENERIC, BIOMED)}
DEFAULT_DOMAIN = CS_AI  # the system ships as a CS/AI research system first


def get_domain(key: str | None) -> Domain:
    if not key:
        return DEFAULT_DOMAIN
    d = _REGISTRY.get(key.strip().lower())
    if d is None:
        raise ValueError(f"unknown domain {key!r}; known: {sorted(_REGISTRY)}")
    return d
