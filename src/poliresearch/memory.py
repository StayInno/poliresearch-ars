"""Memory — Karpathy's LLM-OS memory hierarchy, made concrete.

Karpathy frames an LLM agent like an operating system: the context window is RAM (fast, precise,
tiny), model weights are hazy long-term knowledge, and a real agent needs an explicit, editable,
human-readable memory it actively curates — a "notebook" it distills lessons into ("system-prompt
learning"), rather than an append-only scratchpad that re-amplifies its own hallucinations.

This module implements that hierarchy for the discovery loop:

  working   : ephemeral per-cycle notes               (RAM — cleared on recompile)
  long-term : the corpus + BM25/embedding index        (disk — see retrieval.py)
  notebook  : DISTILLED, curated lessons + findings     (the editable memory the agent writes)

The key operation is `distill()` ("state recompilation"): periodically compress the accumulated
findings/notes into a compact, validated notebook and DISCARD the raw history — which is exactly
the failure-mode cure the discovery engine itself hypothesised (autocatalytic hallucination
amplification in append-only memory).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Memory:
    goal: str = ""
    corpus_hash: str = ""
    working: list[str] = field(default_factory=list)        # RAM (ephemeral)
    notebook: list[str] = field(default_factory=list)       # distilled lessons (curated)
    findings: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    # --- writes ---
    def scratch(self, note: str) -> None:
        if note:
            self.working.append(note)

    def add_finding(self, text: str, verdict: str, cycle: int) -> None:
        self.findings.append({"text": text, "verdict": verdict, "cycle": cycle})

    def add_open_question(self, q: str) -> None:
        if q and q not in self.open_questions:
            self.open_questions.append(q)

    # --- read path (what gets loaded into a prompt: compact, distilled) ---
    def context(self, max_lessons: int = 8, max_questions: int = 8) -> str:
        parts = []
        if self.notebook:
            parts.append("Distilled lessons so far (your curated memory):\n- " +
                         "\n- ".join(self.notebook[:max_lessons]))
        novel = [f["text"] for f in self.findings if f.get("verdict") == "neutral"][:max_questions]
        if novel:
            parts.append("Novel leads already on the table (do NOT repeat; extend or combine):\n- "
                         + "\n- ".join(novel))
        if self.open_questions:
            parts.append("Open questions:\n- " + "\n- ".join(self.open_questions[:max_questions]))
        return "\n\n".join(parts)

    # --- recompilation: compress + discard raw history (Karpathy distillation) ---
    def distill(self, llm, max_lessons: int = 8) -> None:
        """Compress findings + notebook into a tight set of lessons and CLEAR working memory.
        No-op (but still clears RAM) if no LLM is available or the call fails."""
        self.working.clear()  # always drop ephemeral RAM on recompile
        material = self.notebook + [f["text"] for f in self.findings]
        if not material or llm is None or not getattr(llm, "available", False):
            return
        try:
            joined = "\n".join(f"- {m}" for m in material[:40])
            out = llm.complete(
                "You distill an agent's research memory into a few durable, non-redundant "
                "lessons. Merge duplicates, drop noise, keep only what should guide future "
                "hypotheses. Return a JSON array of short strings.",
                f"Goal: {self.goal}\n\nRaw memory:\n{joined}\n\n"
                f"Return at most {max_lessons} distilled lessons.",
                max_tokens=600,
            )
            start, end = out.find("["), out.rfind("]")
            if start != -1 and end != -1:
                items = json.loads(out[start:end + 1])
                lessons = [str(x).strip() for x in items if str(x).strip()]
                if lessons:
                    self.notebook = lessons[:max_lessons]  # replace, not append (recompile)
        except Exception:
            pass  # keep prior notebook on failure

    # --- persistence ---
    def to_dict(self) -> dict[str, Any]:
        return {"goal": self.goal, "corpus_hash": self.corpus_hash, "notebook": self.notebook,
                "findings": self.findings, "open_questions": self.open_questions}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                              encoding="utf-8")
