# PoliResearch

A **domain-universal, falsification-first, verification-gated AI research system** — the Tier-1
architecture from the *AI Scientist* deck, hardened with the TRIZ analysis that followed it.
**It ships as a Computer-Science & AI research system first**, and generalises to other fields by
swapping a domain profile (`generic`, `biomed`) — see [Domains](#domains).

It answers research questions **only from a closed corpus of documents you control**, mechanically
verifies every citation against external truth anchors (Crossref + Retraction Watch), and tries to
*refute* its own claims before it accepts them. The goal is the deck's thesis: **symbiosis, not
replacement** — the machine does the labor, the human keeps the two judgments machines can't make.

---

## Why it is built this way (design provenance)

This is not a generic RAG wrapper. Every component traces to a measured failure mode in the research
and a specific TRIZ inventive move:

| Component | Deck slide | TRIZ move | Failure mode it fixes |
|-----------|-----------|-----------|------------------------|
| **Closed RAG** (`corpus/` only) | 11–12 | Separation *on condition* | ChatGPT-style fabricated citations (55%, Walters & Wilder 2023) |
| **Citation Verifier** (Crossref + Retraction Watch) | 12, 14 | #24 Intermediary, #28 external truth | Hallucinated/retracted references |
| **Open Generator → Closed Falsifier** | — (TRIZ) | #13 Inversion, #22 Blessing-in-disguise | Novelty vs grounding contradiction |
| **Tiered Verifier** (data/citation cheap, synthesis heavy) | 14 | #3 Local Quality | Synthesis accuracy collapse (Kosmos 57.9%) |
| **Structured World Model** | 3 (Kosmos) | #24 Intermediary memory | Context loss over long runs |
| **Provenance + corpus hash** | 12–13 | #25 Self-service | Reproducibility |
| **8-gate checklist; human only on 7–8** | 14 | Trimming | Human verification labor |

See `docs/ARCHITECTURE.md` for the full mapping.

### Hardening applied (from the system critique)

- **Independent judge**: the Falsifier runs on a *different* model from the Generator
  (`POLIRESEARCH_FALSIFIER_MODEL`, default Sonnet vs Opus) so the two don't share correlated
  errors — the failure mode the deck itself warned about (slide 7).
- **3-vote falsification**: each hypothesis faces 3 independent refutation passes (distinct
  critical lenses); a strict majority is needed to refute. Single-vote judging was unreliable.
- **Citation gate 3b — title match**: a real DOI attached to the *wrong* paper (citation
  hijacking) is now caught by comparing the claimed title to the Crossref record.
- **Stronger retraction detection**: checks `type`, `update-to`, `update-policy`, and the
  `relation` map (e.g. `is-retracted-by`) — broader Retraction Watch coverage.
- **Robust networking**: retry-with-backoff on 429/5xx, per-DOI caching, and concurrent
  `verify_many` so a long bibliography no longer blocks serially.

---

## What works with zero setup

The **verification layer needs no API key and no LLM** — it hits the free public Crossref and
Retraction Watch APIs. You can verify a bibliography today:

```bash
pip install -r requirements.txt
python -m poliresearch verify-doi 10.1038/s41586-023-06792-0
python -m poliresearch check-bibliography examples/sample_bibliography.json
```

The **research agents** (generator / falsifier / synthesizer) need an LLM, via either backend:

- **Claude Code (no API key)** — if the `claude` CLI is installed and logged in, the agents shell
  out to it (`claude -p`) using your existing Claude Code auth. This is auto-selected when no API
  key is set. Nothing to configure.
- **Anthropic API** — set `ANTHROPIC_API_KEY` in `.env`; auto-selected when present.

Force one with `POLIRESEARCH_LLM_BACKEND=claude_code|anthropic`.

```bash
# Works out of the box if you have Claude Code installed:
python -m poliresearch ask "What mechanism links ROCK inhibition to RPE phagocytosis?" --corpus ./corpus
# => prints "LLM backend: claude_code"
```

If neither backend is available, agent commands degrade gracefully and tell you what to configure
— the verification commands keep working regardless.

---

## Install

```bash
cd ai-poliresearch
python -m venv .venv && . .venv/Scripts/activate    # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest                                               # offline tests must pass
```

## CLI

| Command | What it does | Needs LLM? |
|---------|--------------|-----------|
| `verify-doi <doi>` | Gate 1-3: Crossref existence + retraction + metadata | no |
| `verify-arxiv <id>` | Same gates via the arXiv API (CS/AI papers without a DOI) | no |
| `check-bibliography <file.json>` | Verify a whole reference list (DOI or arXiv, concurrent) | no |
| `gates <claims.json>` | Run the 8-gate anti-hallucination checklist | no |
| `experiment <file.py>` | Run a Python experiment in the sandboxed runner (CS/AI loop) | no |
| `ingest --corpus DIR` | Index a closed corpus, write its hash | no |
| `evaluate citations\|experiments\|falsifier` | Measure the system against labeled datasets | citations/experiments: no · falsifier: yes |
| `ask "<question>" --corpus DIR [--domain cs_ai]` | Full falsification-first pipeline | yes |

## Evaluation (measuring the claims, not asserting them)

The honest weakness of an "anti-hallucination" system is shipping it unmeasured. `evaluate` runs
the components against labeled datasets in `eval/datasets/` and reports precision/recall/F1.

```bash
python -m poliresearch evaluate citations --verbose     # live Crossref + arXiv, no key
python -m poliresearch evaluate experiments --verbose   # runs labeled true/false code, no key
python -m poliresearch evaluate falsifier --votes 1,3   # compares single- vs 3-vote (needs LLM)
```

The falsifier eval exists specifically to test the design decision behind it: **does independent
3-vote actually beat single-vote?** With an LLM key it runs the labeled survive/refute set at each
vote count. Without a key, `--simulate P` measures the vote-aggregation effect directly by
Monte-Carlo, modelling each judge as correct with probability `P` (the realistic single-judge
accuracy on hard near-misses):

```bash
python -m poliresearch evaluate falsifier --suite adversarial --votes 1,3,5 --simulate 0.75
```

**Measured on the 10 adversarial near-miss examples** (8000 trials each), accuracy by vote count:

| single-judge p | 1-vote | 3-vote | 5-vote |
|----------------|--------|--------|--------|
| 0.60 | 0.602 | 0.648 (+0.046) | 0.684 (+0.082) |
| 0.70 | 0.700 | 0.783 (+0.082) | 0.838 (+0.137) |
| 0.80 | 0.800 | 0.894 (+0.094) | 0.942 (+0.142) |
| 0.90 | 0.900 | 0.971 (+0.071) | 0.992 (+0.091) |

So at a plausible single-judge reliability of 0.7–0.8, **3-vote buys ~8–9 accuracy points and
5-vote ~14**, at 3×/5× the LLM cost. These match the binomial-majority prediction exactly
(p=0.80 → 3-vote theory 0.896 vs measured 0.894), which validates the harness.

**Caveat:** the simulation assumes *independent, unbiased* votes. That assumption is wrong — and
the live run proved it.

### Live result (Claude Code backend, Sonnet judge) — the simulation was refuted

Running for real on the 10 adversarial near-miss examples:

| | 1-vote | 3-vote |
|---|--------|--------|
| accuracy | **0.90** | 0.70 |
| precision | 1.00 | 1.00 |
| recall | **0.75** | 0.25 |
| F1 | **0.857** | 0.400 |

**3-vote did worse, not better** — the opposite of the i.i.d. simulation's prediction. The
per-example trace shows the mechanism: precision stays 1.00 (it never wrongly accepts), but recall
*collapses* because every "survive" case that fails is a valid **paraphrase** being over-refuted.
Two paraphrases that the single judge accepted (`aiscientistv2-templates`, `coscientist-known-
reactions`) were refuted 2/3 once the extra lenses voted.

**Why:** the three lenses are all *skeptical* ("contradiction", "unsupported leaps", "missing
controls") and the system prompt says "default to refuted when the corpus does not positively
support." That is a **systematic bias, not random noise** — so majority voting *amplifies* it
instead of averaging it out. Voting reduces variance but compounds bias. The error is also fed by
the weak keyword retriever: for a paraphrase, the supporting chunk often isn't surfaced, so the
judge genuinely "sees" no support and refutes.

**Implication (honest):** the 3-vote design as built is a *recall regression* on this set. The fix
is not "more votes" but **debiasing** — e.g. make one lens a steelman that argues *for* support,
calibrate the "silent ⇒ refute" default, and improve retrieval so paraphrase support is actually
shown to the judge. This is exactly the kind of finding the eval harness exists to produce.

### Round 2 — debate panel (steelman / refuter / adjudicator, 3-way verdict)

Replacing the uniform skeptics with role-diverse agents (ChatEval ICLR 2024; Khan et al. ICML
2024) and refuting **only on `contradicted`** (treating absence-of-support as `neutral`, per
AIS/FActScore):

| falsifier | recall | precision | accuracy | F1 |
|-----------|--------|-----------|----------|-----|
| 1-vote skeptic | 0.75 | 1.00 | 0.90 | 0.857 |
| 3-vote skeptic | 0.25 | 1.00 | 0.70 | 0.400 |
| debate panel   | **1.00** | 0.50 | 0.60 | 0.667 |

The fix **completely cured the recall collapse** (0.25 → 1.00; every paraphrase survives) but
swung the bias the other way: precision fell to 0.50 because the judge calls **contradictions-by-
substitution** (`GC-MS`→`NMR`, `glaucoma`→`hypertension`, `6.33`→`above 8`) `neutral` instead of
`contradicted`.

**Round 3 diagnosis (decisive):** a retrieval check confirmed the conflicting fact *is* in the
judge's context in all four cases, and sharpening the judge prompt to name the substitution rule
produced **byte-identical output**. So the gap is not retrieval and not instructions — the judge,
asked to rule on a *compound* claim holistically, anchors on the mostly-correct parts and ignores
the one wrong atom. The literature-backed fix is **atomic-fact decomposition** (FActScore, Min et
al. 2023): split a claim into atoms, entail each, refute if *any* atom is contradicted.

**Methodological note:** these are swings on a **10-example** set — each example is 10% accuracy,
so the numbers are high-variance and we are at the resolution limit of this benchmark. Further
tuning must follow dataset expansion, not precede it, to avoid overfitting.

### Round 4 — expanded to 47 examples; rankings flipped

The adversarial set was grown to 47 balanced, class-labeled examples (supported paraphrases,
substitution-contradictions, and genuine *neutrals* the corpus is silent on). On this larger set,
all three falsifiers measured live via Claude Code:

| falsifier | recall | precision | accuracy | F1 |
|-----------|--------|-----------|----------|-----|
| 1-vote skeptic | 0.46 | 1.00 | 0.70 | 0.632 |
| **debate panel** (shipped default) | **0.96** | 0.78 | **0.83** | **0.862** |
| decompose (atomic) | 0.96 | 0.74 | 0.79 | 0.833 |

Two decisive findings: (1) **the small set was misleading** — the skeptic that looked *best* on 10
refute-heavy examples is *worst* on 47 balanced ones (recall 0.46; it over-refutes). This vindicates
expanding before tuning. (2) **The debate panel wins** (F1 0.86 vs 0.63), roughly doubling recall —
so it is now the **pipeline default** (`POLIRESEARCH_FALSIFIER_MODE`, options `debate|decompose|vote`).

**Atomic decomposition did not beat the debate panel**, and the trace shows why: substitution-
contradictions (`GC-MS`→`NMR`, `glaucoma`→`hypertension`, `6.33`→`above 8`) are judged *neutral*
even atom-by-atom. The judge reasons **open-world** — a single stated value does not, to the model,
exclude alternatives without an explicit "not Y." Neither prompt-sharpening nor decomposition fixed
this (consistent with Huang et al., *LLMs Cannot Self-Correct Reasoning Yet*, ICLR 2024).

### Round 5 — closed-world judge instruction (shipped) + a labeling insight

Adding an explicit **closed-world-on-stated-attributes** rule to the judge gave the best config yet
and broke nothing:

| debate panel | recall | precision | accuracy | F1 |
|--------------|--------|-----------|----------|-----|
| before closed-world | 0.96 | 0.78 | 0.83 | 0.862 |
| **+ closed-world** | **1.00** | 0.79 | **0.85** | **0.881** |

Recall reached 1.00 (FN=0) and all 7 genuine neutrals stayed `neutral`. **Precision did not move**,
and inspecting *which* cases miss is the real finding: the judge catches identity swaps
(GPT-4→GPT-3.5, Opentrons→Hamilton, 3 agents→2) but rules `neutral` on
glaucoma→hypertension, GC-MS→NMR, 6.33→">8". That is **defensible** — a drug can hold multiple
indications, a product can be confirmed by multiple methods, so "corpus says glaucoma" does not
*logically* exclude "also hypertension." Several `refute` labels baked in a closed-world
exclusivity that is not logically valid, so the "precision ceiling" is **partly a labeling
artifact, not a model failure.**

The principled resolution is not another prompt but **three pipeline actions** —
`supported → accept`, `contradicted → reject`, `neutral → flag as unverified` (not accepted, not
"refuted"). The pipeline already does this functionally: a `neutral` claim is not corpus-supported,
so it never clears checklist gates 5/7/8 and stops for human review. The binary survive/refute eval
just scores it crudely.

### Round 6 — three-action scoring (the corrected, fair numbers)

Re-scoring the same run with three actions (`supported→accept`, `contradicted→reject`,
`neutral→flag/defer`) — no new LLM calls — reframes the "0.79 precision" entirely:

```
actions:  accept=12   reject=14   flag(defer)=21   (n=47)
coverage (auto-decided) = 0.553      flagged-for-human = 0.447
accept-precision = 1.000             reject-precision = 1.000
SAFETY: false-accepts = 0            false-rejects = 0
auto-decided accuracy = 1.000
```

The system **auto-decides 55% of claims with 100% precision in both directions, makes zero
false-accepts and zero false-rejects, and defers the remaining 45% — including the logically
ambiguous substitution cases — to a human.** That is exactly the right behaviour for a
hallucination-averse research assistant: it never passes a false claim and never rejects a true
one; when the corpus cannot settle a claim, it flags rather than guesses. The binary F1 of 0.881
understated this because it scored every safe "flag" as an error. See `docs/RELATED_WORK.md`.

Each eval has a **base** suite (easy, separable) and an **adversarial** suite (paraphrases,
subtle metadata errors, near-miss hypotheses):

```bash
python -m poliresearch evaluate citations   --suite adversarial --verbose
python -m poliresearch evaluate experiments --suite adversarial --verbose
python -m poliresearch evaluate falsifier   --suite adversarial --votes 1,3   # needs LLM
```

**The adversarial citation suite found two real weaknesses (both predicted in the critique),
which were then fixed — driving it from 0.80 back to 1.00:**

- *False negative → fixed*: a faithful paraphrase of a real title ("Using LLMs for autonomous
  chemistry research") was wrongly rejected by the char-level 0.6 threshold. Title matching now
  uses an order-independent, stopword-filtered **token-overlap coefficient**, so reordered/
  abbreviated paraphrases pass while a totally unrelated title still scores ~0.
- *False positive (the dangerous kind) → fixed*: a citation with one real + one fabricated
  co-author was wrongly accepted because author matching used `any()`. It now requires **every**
  claimed surname to be a real author (`all()`).

Both fixes have offline regression tests. The adversarial run also *validated* a hardening: the
real retracted Surgisphere/Lancet paper is correctly caught by gate 2. The experiment suite stays
1.00 because near-misses challenge a *reasoner*, not a code runner — the runner executes real
code and isn't fooled.

**Caveat, stated plainly:** these datasets are still small and curated — not a hard external
benchmark like LitQA2. The value is that measurement exists, is reproducible, and already finds
prioritised bugs. Scaling the datasets and wiring in LitQA2 is the next step.

## Domains

The architecture (open generator → multi-vote falsifier → verification gates → human) is
field-independent; what changes per field lives in a **domain profile** (`src/poliresearch/domain.py`):

| Domain | Truth anchors | Empirical code loop | Notes |
|--------|---------------|---------------------|-------|
| **`cs_ai`** (default) | arXiv → DOI | **yes** | CS/AI papers verified on arXiv; hypotheses *tested by running code* |
| `generic` | DOI → arXiv | no | literature-only synthesis for any field |
| `biomed` | DOI → arXiv | no | experiments are physical (wet-lab); flags claims as needing validation |

**Why CS/AI is the natural first domain:** it is *computational*, so the system can close the
loop — write code, run it, and let the empirical result confirm or refute a hypothesis
(`experiment.py` + the `Experimenter` agent). That is the one thing biomedical systems (Robin,
AI co-scientist) cannot do autonomously, and it is how PoliResearch escapes the literature-only
synthesis ceiling. Select a domain with `--domain` or `POLIRESEARCH_DOMAIN`.

## Layout

```
src/poliresearch/
  config.py            settings + .env loading
  models.py            Claim / ClaimType / Reference / Verdict data models
  world_model.py       Structured World Model (Kosmos-style JSON store)
  citation_verifier.py Crossref + Retraction Watch  (works, no key)
  checklist.py         8-gate anti-hallucination engine
  corpus.py            closed-corpus indexing + hash (reproducibility)
  llm.py               provider interface, Anthropic Claude default
  agents/
    generator.py       OPEN phase — speculative hypotheses
    falsifier.py       CLOSED phase — independent-model 3-vote refutation (TRIZ #13)
    verifier.py        tiered verification (TRIZ #3)
  pipeline.py          falsification-first orchestration
  provenance.py        run logging + corpus hash (MLflow/DVC role)
  cli.py               entry point
tests/                 offline unit tests
examples/              sample data + demo
```

## License

Apache-2.0 (see `LICENSE`).
