# Findings memo — LLM token / inference-cost optimization

Consolidated candidate hypotheses from two PoliResearch discovery runs on the question:
*"How can token usage and inference cost in LLMs be reduced without degrading output quality?"*

**Status discipline (read first).** Everything below is a **falsification-survived candidate lead, not a
result.** "Survived" means the debate-panel falsifier did not *contradict* it against a closed corpus
(absence of contradiction — a weak signal), and `--perturb` confirmed the verdict is stable under
paraphrase. Every number (>30%, ≥50%, >70%, <1.3×) is the *hypothesis's own predicted* effect and the
thing to measure — **nothing has been run on real models.** Treat this as a prioritized experiment
backlog, not evidence.

## Provenance

| Run | Corpus | Pipeline | Output |
|-----|--------|----------|--------|
| **Run 1** | 80 papers (KV-cache + quantization, hash `b872ccb8`) | debate panel + bridge-steering + perturb | 8 candidates |
| **Run 2** | 230 papers (Run-1 corpus **+** speculative decoding, serving/batching, reasoning-token budget, retrieval, routing; hash `5cb19095`) | debate panel + bridge-steering + rotating re-injection + perturb | 7 candidates |

Run 2's corpus was expanded specifically to cover the sub-areas an independent completeness critic
flagged as missing from Run 1 (serving, decode-side, reasoning-budget, retrieval). The expansion worked:
the new cross-area bridges below could not have come from Run 1's KV/quant-only corpus.

An independent verification cross-check (an 8-expert Workflow using **uniform 3-vote skeptics** instead
of the debate panel) refuted **24/24** of its own generated hypotheses — empirically re-confirming this
project's core falsifier finding that uniform skeptics over-refute and role-diverse debate (refute-only-
on-contradicted) is the correct design. So the candidates here come from the validated generator.

---

## Tier 1 — HIGH CONFIDENCE (surfaced independently in BOTH runs)

These appeared in Run 1's KV/quant corpus *and* Run 2's full-stack corpus — cross-validated by
re-derivation from different inputs.

### H1. LLM-QAT *reshapes* activations → better flash working-set residency
- **Bridge:** LLM-QAT × LLM-in-a-flash.
- **Claim:** data-free 4-bit QAT pushes pre-activations away from the ReLU/SiLU threshold, increasing the
  fraction of reliably-zero neurons and the cross-token stability of which neurons fire — exactly the
  predictable sparsity flash-offload exploits. So QAT *synergizes* with flash rather than just shrinking
  weights.
- **Predicted effect:** at matched perplexity, 4-bit QAT shows >25% higher cross-token neuron-activation
  overlap and >30% less weight I/O per token than post-training quant (GPTQ/AWQ); advantage grows as
  bit-width drops.
- **Cheapest falsifier:** predictor recall@k and bytes-loaded/token on matched QAT vs PTQ checkpoints.

### H2. Energy (joules/token) re-ranks KV techniques — a *limit / negative* finding
- **Bridge:** KIVI / CacheBlend × a sustainable-NLP energy benchmark.
- **Claim:** per-channel KV quantization and selective recompute trade DRAM traffic for compute, so the
  energy-optimal operating point differs from the memory-optimal one; the two Pareto frontiers cross.
- **Predicted effect:** 2-bit KV gives a *smaller* fractional energy reduction than memory reduction, and
  below some batch-size threshold KIVI/CacheBlend **increase** joules/token despite cutting peak memory.
- **Cheapest falsifier:** jointly instrument CPU/DRAM vs SSD/compute wattage across sparsity and batch size.
- **Why it matters:** a guardrail — "memory savings ≠ cost savings"; the headline metric hides the bill.

---

## Tier 2 — NEW TIER (Run 2 only; unlocked by the corpus expansion)

Cross-stack bridges connecting the newly-added areas (reasoning, speculative decoding, routing) to the
KV/quant cluster. Genuinely novel; not re-derivations.

### N1. Speculative decoding × reasoning — deliberation tokens are more draftable  ⭐ test first
- **Bridge:** Ouroboros / CTC-draft × s1 / chain-of-thought budget-forcing.
- **Claim:** reasoning chains contain long low-entropy, template-like "filler" spans, so phrase-level
  drafting accepts longer drafts inside `<think>` regions than in final-answer text; a CTC blank-collapsing
  draft head is the principled generator for these variable-length chunks.
- **Predicted effect:** mean accepted draft length **≥50% higher** in the reasoning region than the answer
  region of the same generation; the gap **collapses toward zero as temperature rises >~1.0** (a built-in
  falsification control). Combining CTC + Ouroboros gives >20% extra wall-clock at equal output dist.
- **Cheapest falsifier:** instrument accept-length on *existing* CoT traces with a temperature sweep — no
  training, no new system. **This is the recommended first experiment of the whole memo.**

### N2. One confidence signal jointly controls reasoning-budget AND KV-precision
- **Bridge:** ConCISE step-confidence × s1 test-time scaling × KIVI.
- **Claim:** a per-step confidence computed once drives both knobs — low-confidence steps get more thinking
  tokens *and* full KV precision; high-confidence steps are truncated *and* 2-bit quantized.
- **Predicted effect:** >30% combined token+memory reduction at equal accuracy vs applying ConCISE and KIVI
  independently; savings are sub-additive iff confidence genuinely governs both (directly measurable).
- **Cheapest falsifier:** correlate per-step confidence with both the safe truncation point and the safe KV
  bit-width.

### N3. Routing difficulty signal × flash sparsity — a shared locality budget
- **Bridge:** OrchestraLLM small/large routing × LLM-in-a-flash.
- **Claim:** the per-request difficulty signal that routes easy turns to the small model also predicts FFN
  neuron-activation locality, so easy requests are served from a small flash-resident working set.
- **Predicted effect:** small-routed requests show >2× higher cross-token neuron-activation overlap than
  hard-routed; a routed-flash system cuts average weight I/O per token by >40% vs flash alone.
- **Cheapest falsifier:** measure activation-overlap by routing decision on existing traces.

### N4. Token-pruning ∩ low-rank KV act on the SAME tokens → sub-additive
- **Bridge:** LazyLLM token-pruning × LoRC low-rank KV.
- **Claim:** tokens LazyLLM defers as low-attention are exactly those whose KV collapses into a low-rank
  subspace, so the two methods target the same population and don't stack multiplicatively.
- **Predicted effect:** LazyLLM-pruned tokens overlap **>70%** with LoRC's lowest-reconstruction-error
  tokens; a combined pipeline yields **<1.3×** the better single method's memory saving (sub-additive).
- **Cheapest falsifier:** pure analysis on LongBench — measure the token-set overlap.

---

## Run-1-only leads (KV/quant cluster; not re-derived in Run 2)

Recorded for completeness; lower priority than the cross-validated and new-tier items.
- **Depth × low-rank KV compose sub-additively** (Layer-Condensed ⟂ LoRC; ~D×R compression, ΔPPL below sum).
- **Coupled-Quant after matrix-decomposition is *antagonistic*** (decomposition strips the cross-channel
  MI coupled-quant needs) — *cheapest pure measurement of all: channel-pair MI before/after, no training.*
- **Per-head "compressibility score"** jointly governs depth-merge and eviction — with a **direct internal
  contradiction** (one hypothesis says rank- and eviction-tolerance *coincide*, another says they *oppose*).
  Resolve as a *single matched per-head correlation experiment*; the sign decides which (if either) lives.
- **PyramidInfer retention ↔ Layer-Condensed depth-merge** (shared pivot-token concentration; predict r<−0.5).
- **ChunkKV boundaries = variance-minimizing recompute unit** for KVPR/CacheBlend (≥2× quality/recompute-FLOP).

---

## The unifying thesis (both runs, sharpened by Run 2)

The recurring, most-testable meta-claim: **the "savings" attributed to independent stacked methods are
actually mediated by a shared latent signal** — confidence, difficulty, token-entropy, or attention-
importance. Therefore most stacks are **sub-additive**, and the interaction sign is **measurable as an
overlap/correlation BEFORE any system is built.** This reframes a large swath of token-optimization work
from "build the combined system and benchmark it" to "measure the latent-signal overlap first, then only
build the stacks whose signals are independent."

## Recommended experiment order (cheapest decisive first)

1. **N1 speculative-in-reasoning** — accept-length in `<think>` vs answer + temperature sweep (no training).
2. **Coupled-Quant/decomposition MI drop** (Run-1 lead) — pure KV-tensor statistics, no training.
3. **N4 LazyLLM ∩ LoRC token overlap** — analysis on LongBench.
4. **Per-head rank↔eviction sign** (Run-1 crux) — one correlation measurement settles a contradiction.

## Caveats
- Single corpus per run, single discovery run each; "survived" = not contradicted, not confirmed.
- Run corpora (abstract-level) are not committed; re-verify each bridged source against its primary paper
  before measuring (the framings of LLM-in-a-flash, LLM-QAT, KIVI, LoRC, MiniCache, KV-Compress,
  PyramidInfer, Coupled-Quant, ChunkKV, KVPR, CacheBlend, Ouroboros/CTC, ConCISE, s1, OrchestraLLM, LazyLLM).
- The "near-lossless N× prompt/context compression as a free lunch" claim was flagged by the completeness
  critic as the field's most over-claimed idea (benchmark-recoverable answers mask verbatim-detail failures;
  the compressor's own cost is often excluded) — distrust headline compression ratios without worst-case,
  task-conditional, end-to-end-cost evaluation.
