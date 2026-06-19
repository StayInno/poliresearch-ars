"""Evaluation harness — measures whether the system's claims hold (critique item #13).

Three measurable questions:
  1. Citation layer: precision/recall of accept/reject vs a labeled set (live, no LLM).
  2. Experiment loop: does running code correctly classify true/false computational claims (no LLM)?
  3. Falsifier: does independent 3-vote beat single-vote on a labeled survive/refute set (needs LLM)?
"""

from .metrics import Metrics
from .dataset import CitationExample, ExperimentExample, FalsificationExample
from .harness import (
    Report,
    ThreeActionReport,
    evaluate_citations,
    evaluate_experiments,
    evaluate_falsifier,
    compare_falsifier,
    simulate_falsifier,
    score_three_action,
    three_action_from_report,
)

__all__ = [
    "Metrics", "Report", "ThreeActionReport",
    "CitationExample", "ExperimentExample", "FalsificationExample",
    "evaluate_citations", "evaluate_experiments", "evaluate_falsifier", "compare_falsifier",
    "simulate_falsifier", "score_three_action", "three_action_from_report",
]
