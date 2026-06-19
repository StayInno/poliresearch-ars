"""Offline tests for the 3-vote falsifier, using a scripted fake LLM (no network)."""

from __future__ import annotations

from poliresearch.agents.falsifier import (DebatePanelFalsifier, DecompositionFalsifier,
                                            Falsifier)
from poliresearch.corpus import Chunk, Corpus


class _ScriptedLLM:
    """Returns queued responses in order; `model` attr lets the pipeline log it."""

    def __init__(self, responses: list[str], model: str = "fake-judge"):
        self.responses = list(responses)
        self.model = model
        self.available = True
        self.calls = 0

    def complete(self, system, prompt, *, max_tokens=1500) -> str:
        self.calls += 1
        return self.responses.pop(0)


def _corpus() -> Corpus:
    return Corpus(root=".", chunks=[
        Chunk("doc.txt#0", "doc.txt", "ROCK inhibition increases RPE phagocytosis via ABCA1."),
    ], corpus_hash="hash")


def _refute(reason="contradicted"):
    return '{"refuted": true, "reason": "%s"}' % reason


def _survive(reason="supported"):
    return '{"refuted": false, "reason": "%s"}' % reason


def test_majority_refutes_two_of_three():
    llm = _ScriptedLLM([_refute(), _refute(), _survive()])
    ref = Falsifier(llm, n_votes=3).attempt("ROCK inhibition helps RPE", _corpus())
    assert ref.refuted
    assert ref.tally == "2/3 refute"
    assert llm.calls == 3


def test_survives_with_only_one_refute():
    llm = _ScriptedLLM([_refute(), _survive(), _survive()])
    ref = Falsifier(llm, n_votes=3).attempt("ROCK inhibition helps RPE", _corpus())
    assert not ref.refuted
    assert ref.tally == "1/3 refute"


def test_unparseable_vote_counts_as_refutation():
    llm = _ScriptedLLM(["garbage no json", _survive(), _survive()])
    ref = Falsifier(llm, n_votes=3).attempt("claim", _corpus())
    # 1 (garbage->refute) + 0 = 1/3 -> survives, but the garbage vote IS counted as refute
    assert ref.tally == "1/3 refute"


def test_grounding_is_populated_from_corpus():
    llm = _ScriptedLLM([_survive(), _survive(), _survive()])
    ref = Falsifier(llm, n_votes=3).attempt("ROCK inhibition increases phagocytosis", _corpus())
    assert "doc.txt#0" in ref.grounding


# --- Debate panel (steelman / refuter / judge, 3-way verdict) ---
class _RoleLLM:
    """Returns the steelman/refuter freely, and a scripted judge verdict (3rd call)."""

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.model = "fake"
        self.available = True
        self.calls = 0

    def complete(self, system, prompt, *, max_tokens=1500) -> str:
        self.calls += 1
        if "adjudicator" in system:  # the judge
            return '{"verdict": "%s", "reason": "scripted"}' % self.verdict
        return "argument text"


def test_debate_neutral_does_not_refute():
    # Absence of support => NEUTRAL => survives (the recall fix: paraphrase not killed).
    ref = DebatePanelFalsifier(_RoleLLM("neutral")).attempt("a paraphrase", _corpus())
    assert ref.refuted is False
    assert ref.verdict == "neutral"
    assert ref.tally == "verdict=neutral"


def test_debate_contradicted_refutes():
    ref = DebatePanelFalsifier(_RoleLLM("contradicted")).attempt("a wrong claim", _corpus())
    assert ref.refuted is True
    assert ref.verdict == "contradicted"


def test_debate_supported_survives():
    ref = DebatePanelFalsifier(_RoleLLM("supported")).attempt("a true claim", _corpus())
    assert ref.refuted is False
    assert ref.verdict == "supported"


def test_debate_unparseable_judge_defaults_neutral():
    llm = _RoleLLM("garbage")  # verdict not in the allowed set -> parse fails
    ref = DebatePanelFalsifier(llm).attempt("x", _corpus())
    assert ref.verdict == "neutral" and ref.refuted is False  # never kill on parse error


# --- Atomic decomposition falsifier ---
class _DecompLLM:
    """First call returns the atom list; second returns per-atom verdicts."""

    def __init__(self, atoms, verdicts):
        self.atoms = atoms
        self.verdicts = verdicts
        self.model = "fake"
        self.available = True
        self.calls = 0

    def complete(self, system, prompt, *, max_tokens=1500) -> str:
        self.calls += 1
        if "atomic factual claims" in system:  # decompose call
            import json
            return json.dumps(self.atoms)
        import json
        return json.dumps([{"atom": a, "verdict": v}
                           for a, v in zip(self.atoms, self.verdicts)])


def test_decompose_refutes_if_any_atom_contradicted():
    # mostly-correct compound claim with one wrong atom -> refuted (the precision fix).
    llm = _DecompLLM(["ripasudil repurposed by Robin", "approved for hypertension", "for dry-AMD"],
                     ["supported", "contradicted", "supported"])
    ref = DecompositionFalsifier(llm).attempt("Robin repurposed ripasudil (approved for "
                                              "hypertension) for dry-AMD", _corpus())
    assert ref.refuted is True
    assert ref.verdict == "contradicted"
    assert "hypertension" in ref.reason


def test_decompose_all_supported_survives():
    llm = _DecompLLM(["a", "b"], ["supported", "supported"])
    ref = DecompositionFalsifier(llm).attempt("compound true claim", _corpus())
    assert ref.refuted is False and ref.verdict == "supported"


def test_decompose_neutral_survives():
    llm = _DecompLLM(["unknown subject"], ["neutral"])
    ref = DecompositionFalsifier(llm).attempt("claim about silent subject", _corpus())
    assert ref.refuted is False and ref.verdict == "neutral"
