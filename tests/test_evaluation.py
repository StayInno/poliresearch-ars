"""Offline tests for the evaluation harness."""

from __future__ import annotations

from poliresearch.corpus import Chunk, Corpus
from poliresearch.evaluation import (Metrics, evaluate_experiments, evaluate_falsifier,
                                     compare_falsifier)
from poliresearch.evaluation.dataset import (FalsificationExample, load_experiments)
from poliresearch.experiment import ExperimentRunner


def test_metrics_math():
    m = Metrics()
    for pred, actual in [(True, True), (True, True), (True, False), (False, False), (False, True)]:
        m.add(pred, actual)
    assert (m.tp, m.fp, m.tn, m.fn) == (2, 1, 1, 1)
    assert abs(m.precision - 2 / 3) < 1e-9
    assert abs(m.recall - 2 / 3) < 1e-9
    assert abs(m.accuracy - 3 / 5) < 1e-9


def test_experiment_eval_on_real_dataset():
    examples = load_experiments("eval/datasets/experiments.json")
    report = evaluate_experiments(ExperimentRunner(timeout_s=15), examples)
    # The dataset is authored so the runner classifies every example correctly.
    assert report.metrics.accuracy == 1.0, report.fmt(verbose=True)
    assert report.metrics.total == len(examples)


def test_adversarial_experiment_suite_is_deterministic():
    # Near-miss hypotheses challenge a *reasoner*, but the runner executes real code, so it
    # still classifies every labeled example correctly (the experiment loop is robust to them).
    examples = load_experiments("eval/datasets/experiments_adversarial.json")
    report = evaluate_experiments(ExperimentRunner(timeout_s=15), examples)
    assert report.metrics.accuracy == 1.0, report.fmt(verbose=True)
    assert report.metrics.total == len(examples) >= 8


def test_adversarial_datasets_load():
    from poliresearch.evaluation.dataset import load_citations, load_falsification
    cites = load_citations("eval/datasets/citations_adversarial.json")
    fals = load_falsification("eval/datasets/falsification_adversarial.json")
    assert len(cites) >= 10 and len(fals) >= 10
    # adversarial citation set must contain both subtle-error invalids and tolerated valids
    assert {c.label for c in cites} == {"valid", "invalid"}
    assert {f.label for f in fals} == {"survive", "refute"}


# --- Falsifier comparison with a scripted LLM (no network) ---
class _ScriptedLLM:
    """Returns refute/survive per hypothesis id keyed by a marker in the prompt."""

    def __init__(self, plan: dict[str, list[bool]]):
        # plan: marker -> list of per-vote 'refuted' booleans (cycled)
        self.plan = plan
        self.model = "scripted"
        self.available = True
        self._counters: dict[str, int] = {}

    def complete(self, system, prompt, *, max_tokens=1500) -> str:
        marker = next((m for m in self.plan if m in prompt), None)
        votes = self.plan.get(marker, [True])
        i = self._counters.get(marker, 0)
        self._counters[marker] = i + 1
        refuted = votes[i % len(votes)]
        return '{"refuted": %s, "reason": "scripted"}' % ("true" if refuted else "false")


def _corpus():
    return Corpus(root=".", chunks=[Chunk("c#0", "c", "marker_supported is true in the corpus.")],
                  corpus_hash="h")


def test_single_vote_vs_three_vote_differ():
    # A hypothesis where one noisy vote refutes but the majority does not:
    # single-vote (sees the first/refute vote) would REFUTE; 3-vote majority SURVIVES.
    examples = [FalsificationExample(id="x", hypothesis="marker_supported claim", label="survive")]
    plan = {"marker_supported": [True, False, False]}  # 1 refute, 2 survive

    one = evaluate_falsifier(_ScriptedLLM(plan), _corpus(), examples, n_votes=1)
    three = evaluate_falsifier(_ScriptedLLM(plan), _corpus(), examples, n_votes=3)

    # n_votes=1 takes the first vote (refute) -> wrongly rejects a should-survive claim
    assert one.results[0].predicted_accept is False
    assert one.metrics.accuracy == 0.0
    # n_votes=3 -> majority survives -> correct
    assert three.results[0].predicted_accept is True
    assert three.metrics.accuracy == 1.0


def test_simulation_shows_voting_beats_single_judge():
    from poliresearch.evaluation import simulate_falsifier
    from poliresearch.evaluation.dataset import load_falsification
    examples = load_falsification("eval/datasets/falsification_adversarial.json")
    m = simulate_falsifier(examples, [1, 3, 5], reliability=0.75, trials=3000, seed=0)
    # single judge ~ reliability; majority voting strictly improves accuracy when p>0.5.
    assert abs(m[1].accuracy - 0.75) < 0.03
    assert m[3].accuracy > m[1].accuracy
    assert m[5].accuracy > m[3].accuracy


def test_simulation_no_gain_at_chance():
    from poliresearch.evaluation import simulate_falsifier
    from poliresearch.evaluation.dataset import FalsificationExample
    ex = [FalsificationExample(id="a", hypothesis="h", label="survive"),
          FalsificationExample(id="b", hypothesis="h2", label="refute")]
    m = simulate_falsifier(ex, [1, 3], reliability=0.5, trials=4000, seed=1)
    # at p=0.5 voting cannot help; both hover near 0.5.
    assert abs(m[3].accuracy - 0.5) < 0.05


def test_three_action_scoring():
    from poliresearch.evaluation import score_three_action
    pairs = ([("survive", "supported")] * 3 +     # correct accepts
             [("survive", "neutral")] * 2 +        # safe flags
             [("refute", "contradicted")] * 4 +    # correct rejects
             [("refute", "neutral")] * 2 +         # safe flags (NOT false accepts)
             [("refute", "supported")] * 1 +       # 1 dangerous false-accept
             [("survive", "contradicted")] * 1)    # 1 false-reject
    r = score_three_action(pairs)
    assert (r.accepted, r.rejected, r.flagged, r.n) == (4, 5, 4, 13)
    assert r.false_accepts == 1 and r.false_rejects == 1
    assert abs(r.accept_precision - 3 / 4) < 1e-9   # 3 of 4 accepts correct
    assert abs(r.reject_precision - 4 / 5) < 1e-9   # 4 of 5 rejects correct
    assert abs(r.coverage - 9 / 13) < 1e-9          # 4 flagged deferred


def test_three_action_neutral_is_safe_not_false_accept():
    from poliresearch.evaluation import score_three_action
    # a should-refute claim ruled neutral must NOT count as a false accept.
    r = score_three_action([("refute", "neutral")])
    assert r.false_accepts == 0 and r.flagged == 1


def test_compare_falsifier_runs_all_vote_counts():
    examples = [FalsificationExample(id="x", hypothesis="marker_supported claim", label="survive")]
    plan = {"marker_supported": [True, False, False]}
    reports = compare_falsifier(lambda: _ScriptedLLM(plan), _corpus(), examples, [1, 3])
    assert set(reports) == {1, 3}
    assert reports[3].metrics.f1 >= reports[1].metrics.f1
