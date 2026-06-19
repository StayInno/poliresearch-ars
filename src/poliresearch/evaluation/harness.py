"""Evaluation runners — turn labeled datasets + the system's components into measured Reports."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from ..agents.falsifier import DebatePanelFalsifier, DecompositionFalsifier, Falsifier
from ..corpus import Corpus
from ..experiment import ExperimentRunner
from ..llm import LLM
from .dataset import CitationExample, ExperimentExample, FalsificationExample
from .metrics import Metrics


@dataclass
class ExampleResult:
    id: str
    predicted_accept: bool
    actual_accept: bool
    detail: str = ""
    verdict: str | None = None   # 3-way: supported | neutral | contradicted (debate/decompose)

    @property
    def correct(self) -> bool:
        return self.predicted_accept == self.actual_accept


@dataclass
class ThreeActionReport:
    """Scores a verifier under three actions instead of binary survive/refute:
      supported   -> ACCEPT, contradicted -> REJECT, neutral -> FLAG (defer to human).
    For a hallucination-averse system the safety metrics are false-accept (a should-reject claim
    accepted) and false-reject (a should-survive claim rejected); a FLAG is always safe."""
    accepted: int = 0
    rejected: int = 0
    flagged: int = 0
    false_accepts: int = 0   # gold = refute, but action = accept (the dangerous error)
    false_rejects: int = 0   # gold = survive, but action = reject
    n: int = 0

    @property
    def coverage(self) -> float:           # fraction auto-decided (not deferred to a human)
        return (self.accepted + self.rejected) / self.n if self.n else 0.0

    @property
    def accept_precision(self) -> float:   # of accepted claims, fraction actually true
        return (self.accepted - self.false_accepts) / self.accepted if self.accepted else 1.0

    @property
    def reject_precision(self) -> float:   # of rejected claims, fraction actually false
        return (self.rejected - self.false_rejects) / self.rejected if self.rejected else 1.0

    def fmt(self) -> str:
        return (
            f"  actions: accept={self.accepted}  reject={self.rejected}  flag(defer)={self.flagged}"
            f"  (n={self.n})\n"
            f"  coverage(auto-decided)={self.coverage:.3f}  "
            f"flagged-for-human={self.flagged / self.n if self.n else 0:.3f}\n"
            f"  accept-precision={self.accept_precision:.3f}  "
            f"reject-precision={self.reject_precision:.3f}\n"
            f"  SAFETY: false-accepts={self.false_accepts}  false-rejects={self.false_rejects}"
        )


def score_three_action(pairs: list[tuple[str, str]]) -> ThreeActionReport:
    """pairs = (gold_label, verdict), gold in {survive, refute}, verdict in
    {supported, neutral, contradicted}."""
    r = ThreeActionReport()
    for gold, verdict in pairs:
        r.n += 1
        gold_refute = (gold == "refute")
        if verdict == "supported":
            r.accepted += 1
            if gold_refute:
                r.false_accepts += 1
        elif verdict == "contradicted":
            r.rejected += 1
            if not gold_refute:
                r.false_rejects += 1
        else:  # neutral -> flag/defer (always safe)
            r.flagged += 1
    return r


def three_action_from_report(rep: "Report") -> ThreeActionReport:
    pairs = [("survive" if r.actual_accept else "refute", r.verdict or "")
             for r in rep.results if r.verdict]
    return score_three_action(pairs)


@dataclass
class Report:
    name: str
    metrics: Metrics = field(default_factory=Metrics)
    results: list[ExampleResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # examples skipped due to LLM/tool failure

    def record(self, r: ExampleResult) -> None:
        self.results.append(r)
        self.metrics.add(r.predicted_accept, r.actual_accept)

    def error(self, example_id: str, msg: str) -> None:
        self.errors.append(f"{example_id}: {msg}")

    def fmt(self, verbose: bool = False) -> str:
        lines = [f"== {self.name} ==", self.metrics.fmt()]
        if self.errors:
            lines.append(f"  ({len(self.errors)} example(s) skipped due to errors)")
        if verbose:
            for r in self.results:
                mark = "ok " if r.correct else "XX "
                lines.append(f"  [{mark}] {r.id}: pred_accept={r.predicted_accept} "
                             f"actual={r.actual_accept}  {r.detail}")
            for e in self.errors:
                lines.append(f"  [ERR] {e}")
        return "\n".join(lines)


# --- 1. Citation verification (no LLM; hits Crossref/arXiv) ---
def evaluate_citations(verifier, examples: list[CitationExample]) -> Report:
    rep = Report("Citation verification")
    checks = verifier.verify_many([e.reference for e in examples])
    for e, chk in zip(examples, checks):
        rep.record(ExampleResult(
            id=e.id, predicted_accept=chk.ok, actual_accept=e.should_accept,
            detail=f"src={chk.source} retracted={chk.retracted} "
                   f"err={chk.error or '-'}",
        ))
    return rep


# --- 2. Experiment loop (no LLM; runs the labeled scripts) ---
def evaluate_experiments(runner: ExperimentRunner, examples: list[ExperimentExample]) -> Report:
    rep = Report("Experiment loop")
    for e in examples:
        res = runner.run_python(e.code)
        rep.record(ExampleResult(
            id=e.id, predicted_accept=res.success, actual_accept=e.should_accept,
            detail=res.verdict_line,
        ))
    return rep


# --- 3. Falsifier (needs an LLM); compares vote counts ---
def evaluate_falsifier(llm: LLM, corpus: Corpus, examples: list[FalsificationExample],
                       n_votes: int = 3, mode: str = "vote", max_workers: int = 6) -> Report:
    if mode == "debate":
        falsifier = DebatePanelFalsifier(llm)
        rep = Report("Falsifier (debate: steelman/refuter/judge, 3-way verdict)")
    elif mode == "decompose":
        falsifier = DecompositionFalsifier(llm)
        rep = Report("Falsifier (decompose: atomic facts, refute if any atom contradicted)")
    else:
        falsifier = Falsifier(llm, n_votes=n_votes)
        rep = Report(f"Falsifier (vote, n_votes={n_votes})")

    def judge(e: FalsificationExample):
        # Returns (example, refutation) or (example, Exception). Each call is an independent
        # subprocess, so examples run concurrently — wall-time ~= (calls / max_workers).
        try:
            return e, falsifier.attempt(e.hypothesis, corpus)
        except Exception as exc:
            return e, exc

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        outcomes = list(pool.map(judge, examples))

    for e, res in outcomes:  # record sequentially (Report is not thread-safe)
        if isinstance(res, Exception):
            rep.error(e.id, str(res)[:160])
            continue
        rep.record(ExampleResult(
            id=e.id, predicted_accept=not res.refuted, actual_accept=e.should_accept,
            detail=f"{res.tally}", verdict=res.verdict,
        ))
    return rep


def compare_falsifier(make_llm: Callable[[], LLM], corpus: Corpus,
                      examples: list[FalsificationExample],
                      votes_list: list[int]) -> dict[int, Report]:
    """Run the falsifier eval at several vote counts to test whether 3-vote beats 1-vote.
    `make_llm` is a thunk so each configuration gets a fresh client/conversation state."""
    return {v: evaluate_falsifier(make_llm(), corpus, examples, n_votes=v) for v in votes_list}


def simulate_falsifier(examples: list[FalsificationExample], votes_list: list[int],
                       reliability: float, trials: int = 4000, seed: int = 0) -> dict[int, Metrics]:
    """Measure the vote-aggregation effect WITHOUT an LLM, by Monte-Carlo simulation.

    Models each independent judge vote as correct with probability `reliability` (the realistic
    single-judge accuracy on hard near-misses). For each example and trial we draw `v` votes,
    take the majority refute/survive decision, and score it against ground truth. This isolates
    exactly what multi-vote buys: when a single judge is better than a coin flip (p>0.5),
    majority voting reduces variance and raises accuracy toward 1 as votes increase.

    It is a model of the *design decision*, not the real falsifier — live LLM numbers need a key
    (`evaluate falsifier` without --simulate). But the curve it produces is the textbook reason
    the 3-vote design exists, measured on this dataset's label distribution."""
    rng = random.Random(seed)
    out: dict[int, Metrics] = {}
    for v in votes_list:
        m = Metrics()
        for e in examples:
            correct_is_refute = not e.should_accept  # ground-truth-correct single decision
            for _ in range(trials):
                refute_votes = sum(
                    1 for _ in range(v)
                    if (correct_is_refute if rng.random() < reliability else not correct_is_refute)
                )
                predicted_refute = refute_votes > v / 2
                m.add(predicted_positive=not predicted_refute, actual_positive=e.should_accept)
        out[v] = m
    return out
