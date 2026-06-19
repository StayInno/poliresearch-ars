"""Experimenter — turns a surviving CS/AI hypothesis into a runnable test (closed loop).

Only meaningful for domains where `enable_experiments` is true (CS/AI). It asks the LLM to write
a small, self-contained, dependency-light Python script that empirically tests the hypothesis,
following the runner's convention: exit non-zero (assert/sys.exit) when the hypothesis is FALSE.
The script is then executed by `ExperimentRunner` and the result feeds the verdict.

This is what lets the system produce evidence the closed corpus does not contain — the escape
from the literature-synthesis ceiling.
"""

from __future__ import annotations

import re

from ..llm import LLM

_SYSTEM = (
    "You are a research engineer. Given a computational hypothesis, write ONE self-contained "
    "Python script that empirically tests it. Rules: use only the Python standard library (plus "
    "numpy if essential); be fast (<20s); print what you measured; and CRUCIALLY exit with a "
    "non-zero status (raise AssertionError or sys.exit(1)) if the hypothesis is FALSE, exit 0 if "
    "it holds. Return ONLY the code in a single ```python fenced block."
)


class Experimenter:
    def __init__(self, llm: LLM):
        self.llm = llm

    def write_experiment(self, hypothesis: str, framing: str = "") -> str | None:
        prompt = (
            f"{framing}\n\nHypothesis to test empirically:\n{hypothesis}\n\n"
            "Write the test script now."
        )
        raw = self.llm.complete(_SYSTEM, prompt, max_tokens=1200)
        return _extract_code(raw)


def _extract_code(raw: str) -> str | None:
    m = re.search(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # if no fence, accept the whole thing only if it looks like code
    if "import " in raw or "def " in raw or "assert " in raw:
        return raw.strip()
    return None
