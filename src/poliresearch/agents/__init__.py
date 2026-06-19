"""The three separated agents that resolve the novelty-vs-grounding contradiction.

TRIZ separation of the central physical contradiction ("closed to corpus" for truth AND
"open beyond corpus" for novelty):

  generator  — OPEN phase, separated in TIME: proposes freely, hallucination allowed as fuel
  falsifier  — CLOSED phase, separated in TIME: tries to REFUTE each hypothesis from the corpus
  verifier   — separated by CLAIM TYPE: cheap checks for data/citation, heavy for synthesis
"""

from .generator import Generator
from .falsifier import Falsifier, DebatePanelFalsifier, DecompositionFalsifier
from .verifier import TieredVerifier
from .experimenter import Experimenter

__all__ = ["Generator", "Falsifier", "DebatePanelFalsifier", "DecompositionFalsifier",
           "TieredVerifier", "Experimenter"]
