"""Binary classification metrics with a confusion matrix.

Convention: the "positive" class is the thing the system should ACCEPT — a trustworthy citation,
a true computational claim, or a hypothesis that should SURVIVE falsification. A false positive is
therefore the dangerous case (the system accepted something it should have rejected), so precision
matters most for a hallucination-averse system.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def add(self, predicted_positive: bool, actual_positive: bool) -> None:
        if predicted_positive and actual_positive:
            self.tp += 1
        elif predicted_positive and not actual_positive:
            self.fp += 1
        elif not predicted_positive and not actual_positive:
            self.tn += 1
        else:
            self.fn += 1

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @staticmethod
    def _div(a: int, b: int) -> float:
        return a / b if b else 0.0

    @property
    def precision(self) -> float:
        return self._div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return self._div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return self._div(2 * p * r, p + r)

    @property
    def accuracy(self) -> float:
        return self._div(self.tp + self.tn, self.total)

    def fmt(self) -> str:
        return (
            f"  accuracy={self.accuracy:.3f}  precision={self.precision:.3f}  "
            f"recall={self.recall:.3f}  f1={self.f1:.3f}\n"
            f"  confusion: TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn} "
            f"(n={self.total})"
        )
