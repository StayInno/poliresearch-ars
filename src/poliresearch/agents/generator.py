"""Generator — the OPEN phase (TRIZ #22 Blessing in Disguise).

Deliberately corpus-unbounded: it proposes candidate hypotheses/answers freely, because
constraining novelty at generation time is what ceilings systems at literature re-synthesis.
Whatever it invents is treated as an *unverified seed* — it is never output directly; it must
survive the Falsifier and the checklist first. This is how hallucination becomes fuel instead
of failure.
"""

from __future__ import annotations

import json

from ..corpus import Corpus
from ..llm import LLM

_SYSTEM = (
    "You are a scientific hypothesis generator. Propose candidate answers and mechanistic "
    "hypotheses for the research question. Be bold and specific — novelty is wanted here. "
    "Each hypothesis will be independently fact-checked and an attempt made to refute it, so "
    "do not self-censor, but do make each hypothesis concrete and falsifiable."
)


class Generator:
    def __init__(self, llm: LLM):
        self.llm = llm

    def propose(self, question: str, corpus: Corpus, n: int = 4, framing: str = "") -> list[str]:
        # Provide corpus context as grounding material, but allow the model to go beyond it.
        hits = corpus.keyword_search(question, k=6)
        context = "\n\n".join(f"[{h.chunk_id}] {h.text[:600]}" for h in hits)
        prompt = (
            f"{framing}\n\n" if framing else ""
        ) + (
            f"Research question:\n{question}\n\n"
            f"Relevant corpus excerpts (context, you may also reason beyond them):\n{context}\n\n"
            f"Propose {n} concrete, falsifiable hypotheses or candidate answers. "
            f"Return a JSON array of strings, nothing else."
        )
        raw = self.llm.complete(_SYSTEM, prompt, max_tokens=1200)
        return _parse_list(raw, n)

    def refutable(self, hypotheses: list[str]) -> list[str]:
        """H8: a generation-time abstention filter. For each hypothesis, ask for a CONCRETE
        refutation protocol (the observation/computation that would falsify it). Hypotheses for
        which the model cannot state a falsifier are disproportionately the unverifiable/
        hallucinated ones, so we drop them before spending verification compute on them."""
        if not hypotheses:
            return []
        numbered = "\n".join(f"{i+1}. {h}" for i, h in enumerate(hypotheses))
        raw = self.llm.complete(
            "For each hypothesis, state the single most concrete experiment, query, observation, "
            "or computation that WOULD falsify it. If no concrete falsifier exists (the claim is "
            "untestable as stated), say exactly 'NONE'. Return a JSON array of strings, one per "
            "hypothesis, in order.",
            f"Hypotheses:\n{numbered}",
            max_tokens=1000,
        )
        protocols = _parse_list(raw, len(hypotheses))
        kept = []
        for h, p in zip(hypotheses, protocols):
            if p and p.strip().upper() != "NONE" and len(p.strip()) > 8:
                kept.append(h)
        return kept or hypotheses[:1]  # never abstain on everything; keep one to make progress


def _parse_list(raw: str, n: int) -> list[str]:
    raw = raw.strip()
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1:
        try:
            items = json.loads(raw[start:end + 1])
            return [str(x) for x in items][:n]
        except json.JSONDecodeError:
            pass
    # fallback: line-split
    return [ln.strip("-* ").strip() for ln in raw.splitlines() if ln.strip()][:n]
