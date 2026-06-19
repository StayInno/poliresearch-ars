"""PoliResearch — a falsification-first, verification-gated AI research system.

Tier-1 architecture from the AI Scientist deck, hardened with the TRIZ analysis:
closed-corpus RAG, mechanical citation verification, and an open-generator /
closed-falsifier loop that tries to refute claims before accepting them.
"""

__version__ = "0.1.0"

from .models import Claim, ClaimType, Reference, GateResult, Verdict

__all__ = ["Claim", "ClaimType", "Reference", "GateResult", "Verdict", "__version__"]
