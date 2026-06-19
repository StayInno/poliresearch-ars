# Architecture & design provenance

PoliResearch is built backwards from the research findings: each measured failure mode in the
2024–2026 literature on AI research systems is answered by a specific TRIZ inventive move, and
each move is a concrete module.

## The central contradiction it resolves

> The system must be **closed to its corpus** (to be truthful) **and open beyond its corpus**
> (to be novel).

Closed RAG eliminates fabricated citations but ceilings a system at re-synthesising known
literature (why the AI co-scientist's strongest result merely *recapitulated* a known mechanism).
Opening the system restores novelty but reintroduces hallucination.

**Resolution by separation:**

| Separation | Where | Implementation |
|------------|-------|----------------|
| in **time** | generate openly, *then* verify against corpus | `pipeline.py`: Generator phase → Falsifier phase |
| in **space** | different agents, different epistemic permission | `agents/generator.py` (open) vs `agents/falsifier.py` (closed) |
| on **condition** | open for hypotheses, closed for facts | `models.ClaimType`: synthesis vs data/citation routing |

## Data flow

```
corpus/ ──load_corpus──▶ Corpus (+ sha256 hash, gate 6)
                              │
                    WorldModel (goal, entities, claims, open questions)
                              │
              ┌───────────────┴────────────────┐
   [OPEN]  Generator                            │
   proposes N hypotheses (corpus-aware,          │
   not corpus-bound)  ── TRIZ #22 ──────────────▶ each hypothesis
                                                  │
   [CLOSED] Falsifier ── TRIZ #13 ──────────────▶ tries to refute from corpus only
                                                  │  (automates gate 7)
                              refuted ──▶ logged & dropped
                              survived ──▶ TieredVerifier ── TRIZ #3
                                                  │
                                   8-gate checklist (checklist.py)
                                   gates 1–6 mechanical · 7 auto+human · 8 human
                                                  │
                              accepted? ──▶ WorldModel + provenance JSONL
```

## Why "falsification-first" beats "support-first"

A support-seeking verifier is biased toward confirmation — it asks "can I find evidence *for*
this?" and usually can. The Falsifier inverts the burden (TRIZ #13): it must fail to refute a
claim for the claim to survive. The corpus's own contradictions (PaperQA2 measured ≈2.34 per
paper) are the ammunition. This is the single highest-leverage change versus the systems surveyed.

## Where accuracy is known to fail, and what we do about it

Kosmos reported ~85% accuracy on data claims but only ~57.9% on synthesis claims. So the
`TieredVerifier` routes **synthesis** claims (the dangerous, high-value kind) through the extra
falsification pass, while **data/citation** claims take the cheap mechanical path. Verification
budget follows risk (TRIZ #3, Local Quality) instead of being spread uniformly.

## What is *not* automated (by design)

Gate 8 (domain-expert significance review) never auto-passes, and gate 7 requires human
confirmation even after the Falsifier runs. This is the deck's "human in the loop — mandatory"
and the TRIZ trimming result: machines take gates 1–6; humans keep only the two judgments
(counter-evidence sufficiency, real-world significance) that the research shows machines cannot
yet make reliably.

## Extension points

- **Retrieval**: `corpus.Corpus.keyword_search` is intentionally trivial. Replace with PaperQA2
  or an embedding index without touching the pipeline.
- **LLM provider**: implement `llm.LLM.complete`; everything else is provider-agnostic.
- **Tracking**: drop in MLflow (already wired in `provenance.py`) for full experiment tracking.
