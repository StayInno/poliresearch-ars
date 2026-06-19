"""Tests for the code-execution experiment runner (the CS/AI closed loop)."""

from __future__ import annotations

from poliresearch.experiment import ExperimentRunner


def test_passing_experiment():
    # Hypothesis holds -> exit 0 -> success.
    res = ExperimentRunner(timeout_s=15).run_python(
        "assert sum(range(10)) == 45\nprint('ok')"
    )
    assert res.ran and res.success
    assert "ok" in res.stdout
    assert "PASSED" in res.verdict_line


def test_failing_experiment():
    # Hypothesis false -> assertion -> non-zero exit -> failure.
    res = ExperimentRunner(timeout_s=15).run_python("assert 1 == 2, 'hypothesis false'")
    assert res.ran and not res.success
    assert res.returncode not in (0, None)
    assert "FAILED" in res.verdict_line


def test_timeout():
    res = ExperimentRunner(timeout_s=1).run_python("import time; time.sleep(5)")
    assert res.timed_out
    assert not res.success
    assert "TIMED OUT" in res.verdict_line
