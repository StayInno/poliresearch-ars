# Self-derived improvements

These five upgrades were **proposed by PoliResearch itself**: running the discovery engine on a
90-paper corpus about *autonomous AI research systems* (goal: "how to make such systems more
reliable and capable of genuine novel discovery") produced 8 candidate hypotheses, several of
which mapped directly onto the system's own weaknesses. The High-value ones were then implemented.

| ID | Hypothesis (system-generated) | Implementation | Where |
|----|-------------------------------|----------------|-------|
| **H2** | "Citation-reasoning decoupling": a real source is cited but the inferential link is fabricated → require a supporting **sentence span** and check it entails the claim | Gate 5b: `_span_supports()` checks the grounding span lexically entails the claim (NLI/LLM is the documented upgrade) | `checklist.py`, `models.Claim.grounding_spans` |
| **H4** | Inter-agent disagreement under conflicting priors is a calibrated abstention signal (≫ self-confidence) | Debate falsifier re-adjudicates under "lean SUPPORTED" vs "lean CONTRADICTED" priors; disagreement → `neutral`/abstain | `agents/falsifier.py` (`conflicting_priors`) |
| **H5** | Hallucination concentrates in **numeric** claims (~3× refutation) | `has_numeric_claim()`; `TieredVerifier` routes numeric claims to the strict falsification path regardless of type | `models.py`, `agents/verifier.py` |
| **H7** | **Temporal rediscovery** (freeze corpus at year X, rediscover X+1..X+3 findings) is a manipulation-resistant novelty metric | `split_corpus_by_year()` + `rediscovery_rate()` benchmark | `evaluation/temporal.py` |
| **H8** | Inability to pre-register a **refutation protocol** is a strong hallucination signal (AUC>0.75 vs self-confidence <0.6) | `Generator.refutable()` drops hypotheses for which no concrete falsifier can be stated, *before* verification | `agents/generator.py`, wired in `discovery.py` |
| **H3** | Novelty concentrates in **cross-corpus bridges**: pairs sharing no authors/citations at *intermediate* distance (inverted-U) | `bridge_pairs()` selects no-shared-author/no-shared-ref pairs in a mid-band similarity, steers generation toward them | `bridges.py`, wired in `discovery.py` (`bridge_steering`, on by default) |

Notably, **H2 is exactly the gate-5 grounding gap** flagged in the original system critique, and
H1/H4 re-validate the generator/verifier-separation and role-diversity choices already made — i.e.
the system independently re-derived both its real weaknesses and its correct decisions.

Defaults: H2/H5/H8 are always on (cheap, strict-by-default). H4 (`conflicting_priors`) is opt-in
in `DiscoveryEngine` because it ~doubles judge calls — stage it after the cheaper signals, as the
system's own synthesis recommended. H7 is a benchmark you run, not a runtime change.

Still open (lower value/effort ratio, not yet built): H6 (claim-DAG betweenness-weighted
verification budget), H1-full (auto-route executable sub-claims).
