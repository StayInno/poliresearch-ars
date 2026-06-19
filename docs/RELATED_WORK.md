# Related work — how PoliResearch's results correspond to the literature

PoliResearch's verification findings are, in several places, an independent replication of
published results; in one place they extend the literature with a less-documented negative result.

## Replications

**Role diversity beats uniform agents.** Our debate panel (steelman/refuter/adjudicator, F1 0.86)
decisively beats a uniform 3-skeptic majority (F1 0.40) and a single skeptic (0.63). This
reproduces:
- **ChatEval** — Chan et al., *Towards Better LLM-based Evaluators through Multi-Agent Debate*,
  ICLR 2024 (arXiv:2308.07201): identical-role panels ≈ a single judge; diverse roles +6.2 pts.
- **Khan et al.**, *Debating with More Persuasive LLMs Leads to More Truthful Answers*, ICML 2024
  (arXiv:2402.06782): FOR/AGAINST + judge 76% vs 54% single advocate.
- Multi-agent debate: **Du et al.** ICML 2024 (arXiv:2305.14325); **Liang et al.** EMNLP 2024
  (arXiv:2305.19118, "Degeneration-of-Thought").

**Voting fixes variance, not bias** (why 3-vote < 1-vote). Our uniform skeptics shared a
systematic refute-bias, which majority voting amplified — the Condorcet reversal under correlated,
biased voters.
- **Dietrich & Spiekermann**, "Jury Theorems," SEP 2021 (asymptotic fallibility when p<0.5).
- **Kim et al.**, *Correlated Errors in LLMs*, ICML 2025 (arXiv:2506.07962): same-model errors are
  highly correlated.
- **"More Agents Is All You Need"** (arXiv:2402.05120): sampling-and-voting cancels only
  *non-systematic* error.
- **Wang et al.**, *Self-Consistency*, ICLR 2023 (arXiv:2203.11171): majority over *diverse* paths.

**Verification/synthesis is the weakest link** — matches the flagship "AI scientist" systems.
- **Kosmos** (arXiv:2511.02824): 85% on data claims but **57.9% on synthesis claims**.
- **PaperArena** (arXiv:2510.10909): 38.78% agent vs 83.5% human on cross-paper reasoning.
Our falsifier is a synthesis step and its ceiling sits in the same regime.

**Small/curated benchmarks invert rankings.** Our 10→47 reversal (the skeptic looked *best* on 10
refute-heavy examples, *worst* on 47 balanced) mirrors the deep-research finding that PaperQA2's
"superhuman synthesis" and "RAG-QA SOTA" claims did not survive verification, and that PaperArena's
human baseline was n=3. The field's methodological warning, reproduced.

**Grounding primitives.** Our `supported|neutral|contradicted` verdict and atomic decomposition
follow **AIS** (Rashkin et al., Computational Linguistics 2023, arXiv:2112.12870), **FActScore**
(Min et al., EMNLP 2023, arXiv:2305.14251), and **RAGAS** (Es et al., EACL 2024, arXiv:2309.15217).
LLM-judge bias framing: **Zheng et al.**, NeurIPS 2023 (arXiv:2306.05685).

## Extension (our negative result)

**Atomic decomposition did *not* fix substitution-contradictions.** Even atom-by-atom, the judge
calls `GC-MS`→`NMR` and `glaucoma`→`hypertension` *neutral*, because it reasons **open-world**: a
single stated value does not, to the model, exclude alternatives without an explicit "not Y." The
FActScore/RAGAS recipe assumes decomposition + NLI suffices; for *closed-corpus* verification it
does not. This is consistent with **Huang et al.**, *LLMs Cannot Self-Correct Reasoning Yet*, ICLR
2024 (arXiv:2310.01798) — the limit did not yield to prompt-level fixes. Our attempted remedy is an
explicit **closed-world-assumption** instruction (treat the corpus as the complete record for any
attribute it states); see the README Evaluation section for whether it moved the precision ceiling.

## Position among named systems

| System | Verification | PoliResearch correspondence |
|--------|-------------|------------------------------|
| PaperQA2 | RCS + contradiction detection + Crossref | our citation verifier mirrors it; we add the falsifier |
| AI Co-Scientist | generate–debate–evolve + Elo (self-referential) | our debate panel is a micro-version; we confirm debate helps but quantify the judge bias their Elo can't see |
| Robin / Kosmos | literature + data agents, human validates | same closed-RAG + human-in-loop stance; we measure where the verifier breaks |
| AI Scientist v2 | produces manuscripts with hallucinated results | our falsifier is the missing gate; our precision ceiling = the field's hallucination-detection ceiling |
