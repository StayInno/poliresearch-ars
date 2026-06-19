"""Tests for the claim-DAG betweenness-weighted verification budget (H6)."""

from __future__ import annotations

from poliresearch.claim_graph import ClaimGraph


def _graph():
    # L1,L2 -> M (load-bearing inference) -> R (root); L3 -> R directly.
    # Both L1->R and L2->R shortest paths pass through M, so M has highest betweenness.
    return ClaimGraph(
        nodes={"L1": "evidence1", "L2": "evidence2", "L3": "evidence3",
               "M": "load-bearing inference", "R": "final conclusion"},
        edges=[("L1", "M"), ("L2", "M"), ("M", "R"), ("L3", "R")],
    )


def test_loadbearing_node_has_highest_betweenness():
    cb = _graph().betweenness()
    assert cb["M"] == max(cb.values())
    assert cb["M"] > cb["L3"]          # L3 bypasses M -> lower centrality


def test_verification_order_prioritizes_loadbearing():
    assert _graph().verification_order()[0] == "M"


def test_budget_reaches_loadbearing_node_first():
    g = _graph()
    # tiny budget (1 of 5 nodes) must still cover the load-bearing M
    plan = g.verify_within_budget(lambda _t: True, budget_fraction=0.2)
    assert plan["most_central"] == "M"
    assert plan["most_central_checked"] is True
    assert "M" in plan["checked"]


def test_failure_surfaced_when_loadbearing_fails():
    g = _graph()
    plan = g.verify_within_budget(lambda text: "load-bearing" not in text, budget_fraction=0.4)
    assert "M" in plan["failures"]      # the corrupting node is caught at low budget


def test_empty_graph_safe():
    plan = ClaimGraph().verify_within_budget(lambda _t: True, budget_fraction=0.5)
    assert plan["checked"] == [] and plan["most_central"] is None
