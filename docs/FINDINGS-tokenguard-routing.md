# Findings memo — TokenGuard self-supervised routing (Task 04 / Result D)

Candidate hypotheses from a PoliResearch discovery run on the TokenGuard open problem:
*"How can an LLM coding agent learn online, per-repo and WITHOUT token logprobs, when to escalate from a
cheap model to a strong one, using only free verification byproducts (test pass/fail, build success, retry
counts), and which signals best predict the cheap model's success gap?"*

**Status discipline.** These are **falsification-survived candidate leads, not measured results** — and
this memo is written to TokenGuard's Prime Directive: nothing here is a number to report, every effect size
is the hypothesis's *own prediction* and the thing to **measure** on a real `claude -p` stream. The
generator respected the hard substrate limit: **no hypothesis uses token logprobs** (impossible on this
substrate); c1-1 explicitly substitutes the free test bitvector for the unavailable logprob signal.

## Provenance
- Corpus: 90 papers (LLM routing / cascades / online learning / verification), hash `dd948794`.
- Pipeline: debate-panel falsifier + bridge-steering + refutability filter + conflicting-priors + perturb.
- Output: 8 candidates, all `neutral`/novel (beyond any single corpus paper), paraphrase-stable.

---

## Maps to Task 04 — which free byproducts carry the routing signal (step 5)

### R1. First-attempt *distinct build/import error-class count* > pass/fail or retry count  ⭐ test first
- **Mechanism:** build-stage errors (missing APIs, wrong signatures, unresolved symbols) are repo-structural
  — they reveal the cheap model lacks the repo's latent dependency map, a deficit the strong model reliably
  closes; test failures by contrast reflect logic the strong model often *also* fails.
- **Predicted effect:** a contextual bandit on first-attempt distinct-error-category counts reaches within
  5% of an oracle escalation policy on per-repo regret, and beats a pass/fail-only bandit by >10% net value.
- **Why first:** cheapest (single attempt, no extra sampling), clearest baseline, and the error-class
  primitive is reused by R4/R7/R8 — validating it de-risks the largest portion of the program.
- **Harness:** `tokenbench/measure_router_e2e.py` + the error-class feature.

### R2. Per-repo test-suite *discriminativeness* dominates a global threshold
- **Mechanism:** a global threshold mis-provisions like a worst-case vLLM pool — it over-escalates in repos
  with weak tests (a green run is uninformative, cheap "success" is illusory) and under-escalates in repos
  with strong tests (a single red test is a high-precision escalate signal).
- **Predicted effect:** per-repo threshold beats best global by >15% net value in the top/bottom
  discriminativeness quartiles; advantage ≈0 on repos with near-identical discriminativeness.
- **Estimate discriminativeness online** via mutation/flakiness sampling.

### R3. Marginal *slope* of tests-fixed-per-retry beats absolute counts
- A plateau by retry 2 (marginal gain < ~1 test) is a stronger escalate trigger than total retry count;
  convergence dynamics distinguish "closing on the solution" from "stuck in a fix-loop." Predicted: slope
  ranks above absolute counts in a fitted predictor's permutation importance.

### R4. Failure-mode *taxonomy*-routed thresholds beat any single scalar
- Cluster each task by which signals fired (build-fail / test-fail / flaky / timeout / retry-oscillation)
  and learn a per-cluster threshold. Predicted: ~5 clusters → >12% higher net value, and cluster-conditional
  success gaps differ >2× (genuine heterogeneity).

### R5. Cross-sample agreement over the verification BITVECTOR — the logprob-free confidence proxy  ⭐⭐
- **The substrate-constraint answer.** Run the cheap model k=3–4× and measure the Hamming dispersion of the
  pass/fail-per-test + build-error-class bitvectors. Intermediate-to-high dispersion = high-entropy region
  near the model's capability frontier, where the strong model's marginal benefit is largest. Substitutes
  the free test vector for the unavailable logprobs.
- **Predicted effect:** a bandit on bitvector dispersion reaches within 5% of oracle regret and beats the
  best single-trajectory feature by >10% at equal sampling budget. (Charge the k-sample cost against gains.)

### R6. Zero-shot per-repo threshold transfer via a byproduct-distribution repo embedding
- Embed a repo from the empirical distribution of its free byproducts (error-class histogram, estimated
  discriminativeness, mean retry slope, flaky fraction); k-NN/regress over known repos to initialize the
  threshold before a single local escalation — killing the **cold-start** that cripples per-repo bandits on
  their first tasks (directly relevant to Task 04's "converge in ≤50 tasks" worry). Predicted: >70% of a
  trained bandit's net value over the first 20 tasks of a held-out repo.

---

## Maps to Task 05 — the untested Result-D corollary, operationalized

### R7. "Strong-also-fails" suppression — model BOTH terms of the gap
- **This *is* the Result-D corollary** ("don't escalate hard-for-both tasks", Δs≈0 because the strong model
  also fails) — but expressed as a **free-signal mechanism**: record, on the escalations performed, which
  byproduct signatures co-occur with *strong-model* failure (irreducible difficulty), and suppress future
  escalation when a new failure matches.
- **Predicted effect:** >15% fewer escalations with <1% absolute accuracy drop; matched signatures show
  strong-model success below the repo's break-even rate.
- **Significance:** converts Task 05's *theoretical* corollary into something measurable from free byproducts
  on a frontier-hard stream (SWE-bench) — the discriminating test the brief calls undischarged.

---

## Cross-thread bridge — Task 04 ↔ Task 03 (routing meets the workload moat)

### R8. Error-class-keyed *intervention ladder* before binary escalation
- Most cheap-model failures are fixable by a rung cheaper than full escalation: **build/import errors →
  inject the resolved dependency/symbol map and retry cheap**; flaky/timeout → rerun or raise timeout;
  clean-build-but-logic-fail → escalate. The error class is a near-free diagnosis of *which* deficit caused
  the failure, and deficits map to differently-priced remedies.
- **The moat connection:** the "inject the dependency map" rung is exactly **supplying private workload
  context W** (Task 03's I(S;W|θ)) as a routing action *below* escalation — unifying the routing and
  workload-moat threads.
- **Predicted effect:** same final accuracy as always-escalate-on-failure at >20% lower cost; >40% of
  build/import failures closed by the dependency-map rung without escalation.

---

## Recommended experiment order (on existing `tokenbench` harnesses)
1. **R1** first-attempt error-class count vs pass/fail — cheapest, foundational, supplies the primitive.
2. **R5** bitvector agreement — the logprob-free confidence signal; pair with R1 in one ablation.
3. **R2** discriminativeness-conditioned thresholds — low-cost test of the conditioning hypothesis.
4. **R7** strong-also-fails suppression — needs a frontier-hard stream (SWE-bench, per Task 05).

R5/R6/R8 incur extra sampling/fingerprinting/injection cost that must be charged against their claimed net
value (TokenGuard's "report exploration cost; net savings after that" rule). Several share the error-class
feature, so evaluate in an ablation that isolates marginal contribution, not in isolation.

## Caveats
- Single corpus, single run; abstract-level; "survived" = not contradicted, not confirmed.
- **Hard dependency (per the brief): a real longitudinal per-repo task stream** — none of R1–R8 is measured
  until that exists; do not report mock/simulated convergence as a result.
- A human should confirm none of these smuggle in an unbuildable signal; the falsifier already declined to
  propose logprob/confidence routing, but verify R5's bitvector is genuinely free in the target loop.
- Bridged sources (FrugalGPT, Agreement-Based Cascading, Hybrid LLM, Universal Model Routing, TRACER,
  Select-then-Route, LLM Bandit) are named inside each hypothesis; re-verify framings against primaries.
