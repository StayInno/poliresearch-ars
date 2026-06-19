"""Registry of keyed scientific-paper sources.

Rule (per design): every keyed source is DISABLED by default. A source becomes enabled only when
its API key environment variable is set AND a connector is implemented. With no key it is simply
skipped and the keyless anchors (Crossref/arXiv/PubMed/OpenAlex) handle the request — no errors.

  Implemented connectors : Semantic Scholar, CORE
  Registered, key-gated   : NASA ADS, IEEE Xplore, Springer Nature, Elsevier/Scopus
                            (connector pending -> stay disabled even with a key, reported clearly)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .core_verifier import CoreVerifier
from .keyed_rest import ADSVerifier, ElsevierVerifier, IEEEVerifier, SpringerVerifier
from .semantic_scholar_verifier import SemanticScholarVerifier


@dataclass
class KeyedSource:
    name: str
    env_var: str
    description: str
    factory: Callable[[str | None], object] | None  # builds a verifier; None = connector pending
    requires_key: bool = True  # False = free & keyless -> enabled by default

    @property
    def key(self) -> str | None:
        return os.environ.get(self.env_var) or None

    @property
    def implemented(self) -> bool:
        return self.factory is not None

    @property
    def enabled(self) -> bool:
        if not self.implemented:
            return False
        return True if not self.requires_key else bool(self.key)

    @property
    def status(self) -> str:
        if not self.implemented:
            return "registered (connector pending)"
        if not self.requires_key:
            return "ENABLED (free, no key)" + ("; key set for higher limits" if self.key else "")
        return "ENABLED (key set)" if self.key else "disabled (needs free key)"

    def build(self):
        return self.factory(self.key) if self.enabled else None


REGISTRY: list[KeyedSource] = [
    KeyedSource("Semantic Scholar", "SEMANTIC_SCHOLAR_API_KEY",
                "Citation graph + TLDRs; DOI/arXiv/title",
                lambda key: SemanticScholarVerifier(api_key=key), requires_key=False),
    KeyedSource("CORE", "CORE_API_KEY",
                "Largest open-access full-text aggregator; DOI/title + full text",
                lambda key: CoreVerifier(api_key=key)),
    KeyedSource("NASA ADS", "ADS_API_KEY",
                "Astrophysics/physics literature",
                lambda key: ADSVerifier(api_key=key)),
    KeyedSource("IEEE Xplore", "IEEE_API_KEY",
                "CS/EE metadata",
                lambda key: IEEEVerifier(api_key=key)),
    KeyedSource("Springer Nature", "SPRINGER_API_KEY",
                "Metadata + Springer OA full text",
                lambda key: SpringerVerifier(api_key=key)),
    KeyedSource("Elsevier/Scopus", "ELSEVIER_API_KEY",
                "Scopus/ScienceDirect (needs institutional entitlement)",
                lambda key: ElsevierVerifier(api_key=key)),
]


def enabled_sources() -> list[KeyedSource]:
    return [s for s in REGISTRY if s.enabled]
