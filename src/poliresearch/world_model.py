"""Structured World Model — Kosmos's load-bearing idea (deck slide 3, TRIZ #24 Intermediary).

Instead of stuffing everything into a context window (which loses early information over a
long run), the system keeps a queryable, typed store of entities, claims, evidence and the
open questions still to resolve. Every agent reads and writes it. Here it is a simple,
inspectable JSON document — the architecture, not the scale, is the point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WorldModel:
    goal: str = ""
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    claims: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    corpus_hash: str = ""

    # --- entities ---
    def upsert_entity(self, name: str, **attrs: Any) -> None:
        self.entities.setdefault(name, {}).update(attrs)

    # --- claims (each carries its own verification verdict over time) ---
    def add_claim(self, text: str, claim_type: str, status: str = "proposed",
                  **meta: Any) -> int:
        self.claims.append({"text": text, "type": claim_type, "status": status, **meta})
        return len(self.claims) - 1

    def set_claim_status(self, idx: int, status: str, **meta: Any) -> None:
        self.claims[idx]["status"] = status
        self.claims[idx].update(meta)

    def accepted_claims(self) -> list[dict[str, Any]]:
        return [c for c in self.claims if c.get("status") == "accepted"]

    # --- open questions drive the next loop iteration ---
    def add_open_question(self, q: str) -> None:
        if q and q not in self.open_questions:
            self.open_questions.append(q)

    def resolve_question(self, q: str) -> None:
        if q in self.open_questions:
            self.open_questions.remove(q)

    # --- persistence ---
    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "corpus_hash": self.corpus_hash,
            "entities": self.entities,
            "claims": self.claims,
            "open_questions": self.open_questions,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WorldModel":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            goal=d.get("goal", ""),
            entities=d.get("entities", {}),
            claims=d.get("claims", []),
            open_questions=d.get("open_questions", []),
            corpus_hash=d.get("corpus_hash", ""),
        )
