"""Falsification-first orchestration — the architecture from the TRIZ analysis, end to end.

Flow (separation in TIME of the novelty-vs-grounding contradiction):

    load corpus  ->  Structured World Model
        |
    [OPEN]   Generator proposes N hypotheses (corpus-aware but not corpus-bound)
        |
    [CLOSED] Falsifier tries to refute each from the corpus only  (TRIZ #13)
        |
    survivors -> TieredVerifier -> 8-gate checklist               (TRIZ #3)
        |
    accepted (pending human gates 7-8) recorded in the World Model + provenance log
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agents import (DebatePanelFalsifier, DecompositionFalsifier, Experimenter, Falsifier,
                     Generator, TieredVerifier)
from .corpus import load_corpus
from .domain import Domain, DEFAULT_DOMAIN
from .experiment import ExperimentResult, ExperimentRunner
from .llm import LLM
from .models import Claim, ClaimType, Verdict
from .provenance import RunLogger
from .reference_verification import UnifiedVerifier
from .world_model import WorldModel


@dataclass
class HypothesisResult:
    text: str
    refuted: bool
    reason: str
    verdict: Verdict | None = None
    experiment: ExperimentResult | None = None


@dataclass
class PipelineResult:
    question: str
    corpus_hash: str
    results: list[HypothesisResult] = field(default_factory=list)
    world_model_path: str | None = None

    @property
    def survivors(self) -> list[HypothesisResult]:
        return [r for r in self.results if not r.refuted]


def _make_falsifier(mode: str, llm: LLM, n_votes: int):
    if mode == "vote":
        return Falsifier(llm, n_votes=n_votes)
    if mode == "decompose":
        return DecompositionFalsifier(llm)
    return DebatePanelFalsifier(llm)  # default: measured best


class Pipeline:
    def __init__(self, llm: LLM, mailto: str | None, runs_dir: str,
                 falsifier_llm: LLM | None = None, n_votes: int = 3,
                 domain: Domain = DEFAULT_DOMAIN, experiment_timeout_s: int = 30,
                 falsifier_mode: str = "debate"):
        # Generator and falsifier use *different* models when falsifier_llm is supplied, so the
        # judge does not share the generator's correlated errors. The debate panel is the default
        # (measured best F1); "vote" and "decompose" remain available for ablation.
        self.domain = domain
        self.generator = Generator(llm)
        self.falsifier = _make_falsifier(falsifier_mode, falsifier_llm or llm, n_votes)
        self.citations = UnifiedVerifier(mailto=mailto)  # routes DOI -> Crossref, arXiv -> arXiv
        self.verifier = TieredVerifier(self.citations, self.falsifier)
        # Empirical closed loop — only for computational domains (CS/AI).
        self.experimenter = Experimenter(llm) if domain.enable_experiments else None
        self.runner = ExperimentRunner(timeout_s=experiment_timeout_s) if domain.enable_experiments else None
        self.runs_dir = runs_dir

    def run(self, question: str, corpus_dir: str, n: int = 4) -> PipelineResult:
        corpus = load_corpus(corpus_dir)
        wm = WorldModel(goal=question, corpus_hash=corpus.corpus_hash)
        log = RunLogger(self.runs_dir)
        log.params(question=question, corpus_hash=corpus.corpus_hash, domain=self.domain.key,
                   generator_model=getattr(self.generator.llm, "model", "?"),
                   falsifier_model=getattr(self.falsifier.llm, "model", "?"),
                   n_votes=self.falsifier.n_votes, experiments=self.domain.enable_experiments,
                   n_chunks=len(corpus.chunks))

        out = PipelineResult(question=question, corpus_hash=corpus.corpus_hash)

        # OPEN phase (domain framing steers sources/claim style)
        hypotheses = self.generator.propose(question, corpus, n=n,
                                             framing=self.domain.prompt_framing)
        log.event("generated", count=len(hypotheses))

        for h in hypotheses:
            idx = wm.add_claim(h, ClaimType.SYNTHESIS.value, status="proposed")

            # CLOSED phase — 3-vote refutation attempt
            ref = self.falsifier.attempt(h, corpus)
            res = HypothesisResult(text=h, refuted=ref.refuted, reason=ref.reason)
            log.event("falsified", hypothesis=h[:120], refuted=ref.refuted, tally=ref.tally)

            if ref.refuted:
                wm.set_claim_status(idx, "refuted", reason=ref.reason)
                out.results.append(res)
                continue

            # Survivor -> tiered verification + checklist
            claim = Claim(text=h, claim_type=ClaimType.SYNTHESIS, grounding=ref.grounding)
            verdict = self.verifier.verify(claim, corpus, human_reviewed=False)
            res.verdict = verdict

            # CS/AI closed loop: empirically TEST the surviving hypothesis by running code.
            # Empirical evidence is stronger than literature falsification and can confirm a
            # claim the closed corpus does not contain (escapes the synthesis ceiling).
            if self.experimenter and self.runner:
                code = self.experimenter.write_experiment(h, framing=self.domain.prompt_framing)
                if code:
                    res.experiment = self.runner.run_python(code)
                    log.event("experiment", hypothesis=h[:120],
                              outcome=res.experiment.verdict_line)

            status = "accepted" if verdict.accepted else "pending_human"
            wm.set_claim_status(idx, status,
                                pending=[g.name for g in verdict.pending_human],
                                experiment=res.experiment.verdict_line if res.experiment else None)
            log.event("verified", hypothesis=h[:120],
                      mechanical_pass=not verdict.failed_gates,
                      pending_human=[g.gate for g in verdict.pending_human])
            out.results.append(res)

        wm_path = f"{self.runs_dir}/{log.run_id}.worldmodel.json"
        wm.save(wm_path)
        out.world_model_path = wm_path
        log.event("done", survivors=len(out.survivors), total=len(out.results))
        return out
