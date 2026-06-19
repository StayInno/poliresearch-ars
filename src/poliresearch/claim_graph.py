"""Claim-dependency DAG + betweenness-weighted verification budget (H6).

An autonomous system's output is not a flat list of claims but a DAG: leaves = retrieved
evidence, internal nodes = inferences, root = final conclusion. A single decoupled high-centrality
("load-bearing") inference silently corrupts every claim downstream of it. So spending a fixed
verification budget by **betweenness centrality** — verify the load-bearing nodes first — catches
far more root-level errors than verifying claims uniformly.

This module computes directed betweenness (Brandes) over the dependency DAG and returns a
centrality-ordered verification plan; `verify_within_budget` checks the most central nodes first
and reports whether the budget reached the load-bearing ones.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import ceil
from typing import Callable


@dataclass
class ClaimGraph:
    # node id -> claim text
    nodes: dict[str, str] = field(default_factory=dict)
    # directed edges (src supports/feeds dst): src -> dst
    edges: list[tuple[str, str]] = field(default_factory=list)

    def _adj(self) -> dict[str, list[str]]:
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for s, d in self.edges:
            adj.setdefault(s, []).append(d)
            adj.setdefault(d, adj.get(d, []))
        return adj

    def betweenness(self) -> dict[str, float]:
        """Directed betweenness centrality (Brandes). High = load-bearing inference."""
        nodes = list(self.nodes)
        adj = self._adj()
        cb = {v: 0.0 for v in nodes}
        for s in nodes:
            stack, pred = [], {w: [] for w in nodes}
            sigma = {w: 0 for w in nodes}
            sigma[s] = 1
            dist = {w: -1 for w in nodes}
            dist[s] = 0
            q = deque([s])
            while q:
                v = q.popleft()
                stack.append(v)
                for w in adj.get(v, []):
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        q.append(w)
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)
            delta = {w: 0.0 for w in nodes}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
                if w != s:
                    cb[w] += delta[w]
        return cb

    def verification_order(self) -> list[str]:
        """Node ids ranked by betweenness (load-bearing first), ties broken by out-degree."""
        cb = self.betweenness()
        outdeg = {n: 0 for n in self.nodes}
        for s, _ in self.edges:
            outdeg[s] = outdeg.get(s, 0) + 1
        return sorted(self.nodes, key=lambda n: (cb[n], outdeg.get(n, 0)), reverse=True)

    def verify_within_budget(self, verify_fn: Callable[[str], bool],
                             budget_fraction: float = 0.5) -> dict:
        """Verify the top-centrality nodes first until the budget (fraction of nodes) is spent.
        Returns which nodes were checked and whether the single most-central node was reached."""
        order = self.verification_order()
        k = max(1, ceil(budget_fraction * len(order))) if order else 0
        checked = order[:k]
        results = {n: verify_fn(self.nodes[n]) for n in checked}
        most_central = order[0] if order else None
        return {
            "checked": checked,
            "skipped": order[k:],
            "results": results,
            "most_central": most_central,
            "most_central_checked": most_central in checked,
            "failures": [n for n, ok in results.items() if not ok],
        }
