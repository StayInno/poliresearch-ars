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
