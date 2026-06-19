"""Tests for the self-improvement features H2/H4/H5/H7/H8 (offline)."""

from __future__ import annotations

from poliresearch.agents.falsifier import DebatePanelFalsifier
from poliresearch.agents.generator import Generator
from poliresearch.checklist import _span_supports, run_checklist
from poliresearch.corpus import Chunk, Corpus
from poliresearch.models import Claim, ClaimType, Reference, has_numeric_claim


# --- H5: numeric-claim detection ---
def test_numeric_detection():
    assert has_numeric_claim("KIRA6 IC50 is 13 nM in KG-1 cells")
    assert has_numeric_claim("reduces tokens by 40%")
    assert has_numeric_claim("a 7.5-fold increase")
    assert not has_numeric_claim("ROCK inhibition upregulates ABCA1")
    assert not has_numeric_claim("published in 2023")          # bare year ignored
    assert Claim(text="cut cost by 40%", claim_type=ClaimType.DATA).is_numeric


# --- H2: span-level grounding (entailment proxy) ---
def test_span_supports_entailment():
    claim = "Ripasudil increases RPE phagocytosis via ABCA1"
    good = "We found ripasudil raised RPE phagocytosis through ABCA1 upregulation."
    bad = "The study discussed unrelated retinal anatomy and imaging methods."
    assert _span_supports(claim, good)
    assert not _span_supports(claim, bad)


def test_gate5b_flags_unsupported_span():
    claim = Claim(text="Ripasudil works via VEGF inhibition", claim_type=ClaimType.SYNTHESIS,
                  references=[Reference(doi="10.1/x")], grounding=["c#0"],
                  grounding_spans=["Ripasudil increases phagocytosis through ABCA1, not VEGF."])
    v = run_checklist(claim, corpus_chunk_ids={"c#0"}, corpus_hash="h",
                      falsification_attempted=True, falsification_survived=True, human_reviewed=True)
    # the span (ABCA1) does not support the VEGF claim -> a gate-5 entailment failure
    assert any(g.gate == 5 and not g.passed and "span" in g.name.lower() for g in v.gates)
    assert not v.accepted


# --- H8: refutability filter drops untestable hypotheses ---
class _RefLLM:
    available = True
    model = "fake"

    def complete(self, system, prompt, *, max_tokens=1500):
        # protocol generator: first hypothesis testable, second untestable
        if "would falsify" in system.lower():
            return '["Measure X under condition Y and check Z", "NONE"]'
        return "[]"


def test_refutability_filter_drops_untestable():
    kept = Generator(_RefLLM()).refutable(["testable hypothesis", "untestable vibe"])
    assert kept == ["testable hypothesis"]


# --- H4: conflicting-priors disagreement -> abstain ---
class _PriorLLM:
    available = True
    model = "fake"

    def complete(self, system, prompt, *, max_tokens=1500):
        if "adjudicator" not in system:
            return "argument"
        if "Lean toward SUPPORTED" in prompt:
            return '{"verdict": "supported", "reason": "x"}'
        if "Lean toward CONTRADICTED" in prompt:
            return '{"verdict": "contradicted", "reason": "x"}'
        return '{"verdict": "supported", "reason": "x"}'  # base verdict


def _corpus():
    return Corpus(root=".", chunks=[Chunk("c#0", "c", "some evidence text")], corpus_hash="h")


def test_conflicting_priors_abstains_on_disagreement():
    f = DebatePanelFalsifier(_PriorLLM(), conflicting_priors=True)
    ref = f.attempt("a borderline claim", _corpus())
    assert ref.verdict == "neutral" and not ref.refuted    # disagreement -> abstain


# --- H7: temporal split + rediscovery scorer ---
def test_temporal_split_and_rediscovery(tmp_path):
    from poliresearch.evaluation import split_corpus_by_year, rediscovery_rate
    (tmp_path / "old.txt").write_text("Title: Old\nYear: 2018\n\nAbstract: KV cache quantization.",
                                      encoding="utf-8")
    (tmp_path / "new.txt").write_text(
        "Title: New\nYear: 2021\n\nAbstract: attention mass graded precision for kv cache quantization.",
        encoding="utf-8")
    train, holdout = split_corpus_by_year(tmp_path, cutoff=2019, window=3)
    assert len(train) == 1 and len(holdout) == 1
    # a candidate matching the holdout finding counts as rediscovered
    res = rediscovery_rate(
        ["attention mass graded precision improves kv cache quantization"], holdout, threshold=0.3)
    assert res["rediscovered"] == 1 and res["rate"] == 1.0
