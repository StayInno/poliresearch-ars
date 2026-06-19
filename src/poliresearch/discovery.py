"""Discovery Engine — Kosmos-style iterative, parallel, world-model-driven discovery.

This is the orchestration that turns the single-pass verifier into a scale discovery system,
mirroring Kosmos's design (structured world model + many parallel rollouts + iterative cycles +
a final cited report):

  for each CYCLE:
      generate R hypotheses (conditioned on the accumulating WORLD MODEL, so cycles build on
        earlier findings and open questions — the iterative loop)
      falsify + verify all R in PARALLEL (debate panel + citation gate)
      keep the fresh survivors; record supported ones as candidate findings, neutral ones as
        open questions for the next cycle
  synthesize a cited report of the candidate discoveries

Scale knobs (cycles x rollouts) go to Kosmos magnitudes; cost and human validation are what a
real run adds — a SUPPORTED finding here is a *candidate* that still requires human validation,
exactly as Kosmos reports (79.4% verified, human validation mandatory).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .agents import DebatePanelFalsifier, Generator
from .agents.falsifier import Refutation
from .corpus import Corpus
from .llm import LLM
from .memory import Memory
from .reference_verification import UnifiedVerifier


@dataclass
class Finding:
    text: str
    verdict: str          # supported | neutral
    grounding: list[str]
    cycle: int

    @property
    def is_novel_lead(self) -> bool:
        # A genuinely novel hypothesis goes BEYOND any single paper, so the corpus cannot entail
        # it (that would make it not novel). Survived falsification + not contradicted + not a
        # mere restatement = a novel, falsifiable candidate discovery to be tested.
        return self.verdict == "neutral"

    @property
    def is_corroborated(self) -> bool:
        # Supported = the corpus already states this; valuable verification, but NOT novel.
        return self.verdict == "supported"


@dataclass
class DiscoveryResult:
    goal: str
    corpus_papers: int
    cycles: int
    findings: list[Finding] = field(default_factory=list)
    report: str = ""
    world_model_path: str | None = None

    @property
    def candidate_discoveries(self) -> list[Finding]:
        """Novel, falsifiable leads: survived falsification, consistent with but BEYOND the
        corpus. These are the discoveries to test (human/experimental validation)."""
        return [f for f in self.findings if f.is_novel_lead]

    @property
    def corroborated(self) -> list[Finding]:
        """Claims the corpus already supports — verification wins, not novel discoveries."""
        return [f for f in self.findings if f.is_corroborated]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())[:200]


class DiscoveryEngine:
    def __init__(self, llm: LLM, mailto: str | None = None, *, max_workers: int = 8,
                 falsifier_llm: LLM | None = None, refutability_filter: bool = True,
                 conflicting_priors: bool = False):
        self.generator = Generator(llm)
        self.refutability_filter = refutability_filter   # H8
        self.falsifier = DebatePanelFalsifier(falsifier_llm or llm,
                                              conflicting_priors=conflicting_priors)  # H4
        self.verifier = UnifiedVerifier(mailto=mailto, max_workers=max_workers)
        self.synth_llm = llm
        self.max_workers = max_workers

    def run(self, goal: str, corpus: Corpus, *, cycles: int = 3, rollouts: int = 6,
            runs_dir: str = "./runs") -> DiscoveryResult:
        mem = Memory(goal=goal, corpus_hash=corpus.corpus_hash)  # Karpathy LLM-OS memory
        seen: set[str] = set()
        out = DiscoveryResult(goal=goal, corpus_papers=len({c.source for c in corpus.chunks}),
                              cycles=cycles)

        for cycle in range(cycles):
            # Read path = the DISTILLED notebook (not raw history): compact, curated context.
            framing = mem.context() + "\n\nPropose NOVEL hypotheses that synthesize across " \
                "multiple papers — connections not stated in any single paper."
            hypotheses = self.generator.propose(goal, corpus, n=rollouts, framing=framing)
            fresh = [h for h in hypotheses if _norm(h) not in seen]
            if self.refutability_filter and fresh:
                fresh = self.generator.refutable(fresh)  # H8: drop untestable hypotheses early

            # Parallel rollouts: falsify each survivor concurrently (Kosmos-style fan-out).
            def process(h: str) -> tuple[str, Refutation]:
                return h, self.falsifier.attempt(h, corpus)

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(pool.map(process, fresh))

            for h, ref in results:
                seen.add(_norm(h))
                if ref.refuted:
                    continue
                verdict = ref.verdict or "neutral"
                out.findings.append(Finding(text=h, verdict=verdict,
                                            grounding=ref.grounding, cycle=cycle))
                mem.add_finding(h, verdict, cycle)
                if verdict == "neutral":
                    mem.add_open_question(h)

            # Recompilation: distill memory into a tight notebook, discard raw history.
            mem.distill(self.synth_llm)

        out.report = self._synthesize(goal, out.candidate_discoveries)
        path = f"{runs_dir}/discovery-{corpus.corpus_hash}.memory.json"
        try:
            from pathlib import Path
            Path(runs_dir).mkdir(parents=True, exist_ok=True)  # was silently failing if absent
            mem.save(path)
            out.world_model_path = path
        except Exception:
            pass
        return out

    def _synthesize(self, goal: str, discoveries: list[Finding]) -> str:
        if not discoveries:
            return "No novel candidate leads survived falsification this run."
        bullets = "\n".join(f"- {d.text}" for d in discoveries[:20])
        prompt = (
            f"Research goal:\n{goal}\n\n"
            f"The following are NOVEL cross-paper hypotheses: each survived an adversarial "
            f"falsification panel (the corpus does not contradict it) but goes beyond what any "
            f"single paper states (so it is not yet confirmed):\n{bullets}\n\n"
            "Write a concise scientific summary (5-8 sentences) of these candidate discoveries, "
            "grouping related ones and naming the most promising to test next. "
            "Be measured; these are falsifiable leads requiring experimental/human validation."
        )
        try:
            return self.synth_llm.complete(
                "You are a scientific writer summarising verified candidate findings.", prompt,
                max_tokens=700)
        except Exception as e:
            return f"(synthesis step failed: {e})"
