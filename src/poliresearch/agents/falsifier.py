"""Falsifier — the CLOSED phase and the system's signature move (TRIZ #13, The Other Way Around).

Most systems verify by *seeking support* for a claim. That biases toward confirmation and lets
plausible-but-wrong claims through. The Falsifier instead actively tries to *refute* each
hypothesis using only the closed corpus — it searches for contradictory evidence (the corpus's
own contradictions, which PaperQA2 measured at ~2.34/paper, become ammunition here). A hypothesis
that survives a genuine refutation attempt is far stronger than one that merely found support.

Two hardening changes over the naive design (from the system critique):

  * 3-vote adversarial verification — N independent refutation attempts, the claim is REFUTED
    if a majority vote to refute (matching the deep-research harness's 2/3 rule). Single-vote
    judging is unreliable.
  * Independent judge model — the pipeline gives the Falsifier a *different* LLM from the
    Generator, so generator and judge don't share correlated errors (deck slide 7: "AI writes ->
    AI reviews -> AI cites, correlated errors not caught"). Same-model self-judging reproduces
    exactly that failure mode.

This automates checklist gate 7 (FVA-RAG falsification search). The human still confirms.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..corpus import Corpus
from ..llm import LLM

_SYSTEM = (
    "You are a skeptical scientific referee. Your job is to REFUTE the given hypothesis using "
    "ONLY the supplied corpus excerpts. Look for contradicting evidence, unsupported leaps, "
    "confounds, and missing controls. Default to 'refuted' when the corpus does not positively "
    "support the hypothesis. Do not use outside knowledge - if the corpus is silent, the "
    "hypothesis is unsupported."
)


@dataclass
class Vote:
    refuted: bool
    reason: str


@dataclass
class Refutation:
    hypothesis: str
    refuted: bool
    reason: str
    grounding: list[str]            # corpus chunk ids the judgement rests on
    votes: list[Vote] = field(default_factory=list)
    verdict: str | None = None      # debate mode: "supported" | "neutral" | "contradicted"

    @property
    def tally(self) -> str:
        if self.verdict is not None:
            return f"verdict={self.verdict}"
        r = sum(v.refuted for v in self.votes)
        return f"{r}/{len(self.votes)} refute"


class Falsifier:
    def __init__(self, llm: LLM, n_votes: int = 3):
        self.llm = llm
        self.n_votes = n_votes

    def attempt(self, hypothesis: str, corpus: Corpus) -> Refutation:
        hits = corpus.keyword_search(hypothesis, k=8)
        grounding = [h.chunk_id for h in hits]
        context = "\n\n".join(f"[{h.chunk_id}] {h.text[:700]}" for h in hits)

        votes: list[Vote] = []
        for i in range(self.n_votes):
            votes.append(self._one_vote(hypothesis, context, lens_index=i))

        n_refute = sum(v.refuted for v in votes)
        majority_refuted = n_refute > len(votes) / 2     # strict majority of 3 -> need 2
        reason = "; ".join(v.reason for v in votes if v.refuted) if majority_refuted \
            else "survived majority falsification"
        return Refutation(hypothesis=hypothesis, refuted=majority_refuted,
                          reason=reason, grounding=grounding, votes=votes)

    def _one_vote(self, hypothesis: str, context: str, lens_index: int) -> Vote:
        # Vary the critical lens per vote so the three attempts are not identical probes.
        lenses = [
            "Focus on DIRECT contradicting evidence in the corpus.",
            "Focus on unsupported logical leaps and missing mechanistic links.",
            "Focus on confounds, missing controls, and over-generalisation.",
        ]
        lens = lenses[lens_index % len(lenses)]
        prompt = (
            f"Hypothesis to refute:\n{hypothesis}\n\n"
            f"Corpus excerpts (your ONLY allowed evidence):\n{context}\n\n"
            f"Refutation lens for this pass: {lens}\n\n"
            'Return JSON: {"refuted": true|false, "reason": "..."}. '
            "Set refuted=false ONLY if the corpus positively supports the hypothesis "
            "and you found no contradicting evidence under this lens."
        )
        raw = self.llm.complete(_SYSTEM, prompt, max_tokens=600)
        return _parse_vote(raw)


def _parse_vote(raw: str) -> Vote:
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            d = json.loads(raw[start:end + 1])
            return Vote(refuted=bool(d.get("refuted", True)), reason=str(d.get("reason", "")))
        except json.JSONDecodeError:
            pass
    # Conservative default: an unparseable vote counts as a refutation.
    return Vote(refuted=True, reason="unparseable vote - defaulting to refuted (conservative)")


# --- Debate-panel falsifier: the literature-backed fix for the 3-vote recall collapse ---
#
# The uniform-skeptic panel failed because (a) all roles shared one skeptical bias (ChatEval:
# identical roles ~= single judge), and (b) it conflated "no evidence" with "contradicted",
# over-refuting valid paraphrases. This panel fixes both:
#   * ROLE DIVERSITY (Khan et al. ICML 2024, ChatEval ICLR 2024): a steelman argues FOR support,
#     a refuter argues AGAINST, an adjudicator weighs both.
#   * THREE-WAY NLI VERDICT (AIS / FActScore / RAGAS): supported | neutral | contradicted, and we
#     REFUTE ONLY ON `contradicted`. Absence of support is `neutral` (unverified), not refuted, so
#     a true-but-paraphrased claim is no longer killed.

_STEELMAN_SYSTEM = (
    "You are an advocate. Using ONLY the supplied corpus excerpts, make the STRONGEST honest case "
    "that the hypothesis is SUPPORTED by the corpus. A faithful paraphrase that preserves meaning "
    "counts as support. Cite the chunk ids you rely on. If there is genuinely no support, say so "
    "in one sentence. Do not use outside knowledge."
)
_REFUTER_SYSTEM = (
    "You are a skeptic. Using ONLY the supplied corpus excerpts, make the strongest honest case "
    "that the hypothesis is CONTRADICTED by the corpus (the corpus states something incompatible) "
    "or contains an unsupported leap. Cite chunk ids. Distinguish 'the corpus contradicts this' "
    "from 'the corpus is merely silent on this'. Do not use outside knowledge."
)
_JUDGE_SYSTEM = (
    "You are a neutral adjudicator. Given the hypothesis, the corpus excerpts, and both the "
    "supporting and opposing arguments, classify the hypothesis into exactly one verdict:\n"
    "  SUPPORTED   - the corpus entails it (faithful paraphrases count as entailment).\n"
    "  CONTRADICTED- the corpus states something incompatible with it.\n"
    "  NEUTRAL     - the corpus says nothing about the claim's subject (insufficient evidence).\n"
    "Two rules that decide the hard cases:\n"
    "1. CLOSED-WORLD ON STATED ATTRIBUTES. Treat the corpus as the authoritative and COMPLETE "
    "record for any attribute it explicitly states. If the corpus states that an attribute has a "
    "value X (detection method = GC-MS; approved indication = glaucoma; average score = 6.33; "
    "number of agents = three; timeline = 2.5 months), then a claim asserting a DIFFERENT value "
    "for that SAME attribute (NMR; hypertension; 'above 8'; two; five years) is CONTRADICTED. You "
    "do NOT need the corpus to say 'not Y' - one stated value excludes the alternatives. A "
    "negation of a stated fact (corpus: manuscripts had hallucinations; claim: free of "
    "hallucinations) is likewise CONTRADICTED.\n"
    "2. NEUTRAL applies ONLY when the corpus says nothing about that attribute at all (e.g. it "
    "states no cost, no funding source, no prompt language). Absence of any statement about the "
    "attribute = NEUTRAL; a conflicting stated value = CONTRADICTED.\n"
    "Do not use outside knowledge.\n"
    'Return JSON: {"verdict": "supported|neutral|contradicted", "reason": "..."}'
)


class DebatePanelFalsifier:
    """Steelman + refuter + adjudicator with a three-way verdict. Same interface as Falsifier."""

    def __init__(self, llm: LLM, judge_llm: LLM | None = None, conflicting_priors: bool = False):
        self.llm = llm
        self.judge_llm = judge_llm or llm
        self.conflicting_priors = conflicting_priors  # H4: disagreement-as-abstention
        self.n_votes = 3  # 3 role calls; kept for logging compatibility

    def attempt(self, hypothesis: str, corpus: Corpus) -> Refutation:
        hits = corpus.keyword_search(hypothesis, k=8)
        grounding = [h.chunk_id for h in hits]
        context = "\n\n".join(f"[{h.chunk_id}] {h.text[:700]}" for h in hits)

        steel = self.llm.complete(_STEELMAN_SYSTEM, self._case_prompt(hypothesis, context),
                                  max_tokens=500)
        against = self.llm.complete(_REFUTER_SYSTEM, self._case_prompt(hypothesis, context),
                                    max_tokens=500)
        verdict, reason = self._adjudicate(hypothesis, context, steel, against)

        # H4: re-adjudicate under conflicting priors; if the two disagree, the claim is
        # uncertain — abstain (NEUTRAL) rather than trust one framing. Disagreement is a far
        # better signal than a single judge's self-confidence.
        if self.conflicting_priors:
            v_pos, _ = self._adjudicate(hypothesis, context, steel, against,
                                        prior="Lean toward SUPPORTED unless the corpus clearly conflicts.")
            v_neg, _ = self._adjudicate(hypothesis, context, steel, against,
                                        prior="Lean toward CONTRADICTED unless the corpus clearly supports.")
            if v_pos != v_neg:
                verdict, reason = "neutral", (f"abstain: verdicts disagree under conflicting "
                                              f"priors ({v_pos} vs {v_neg})")

        refuted = verdict == "contradicted"
        return Refutation(hypothesis=hypothesis, refuted=refuted, reason=reason,
                          grounding=grounding, verdict=verdict)

    @staticmethod
    def _case_prompt(hypothesis: str, context: str) -> str:
        return (f"Hypothesis:\n{hypothesis}\n\n"
                f"Corpus excerpts (your ONLY allowed evidence):\n{context}\n\n"
                "State your case in 3-5 sentences.")

    def _adjudicate(self, hypothesis: str, context: str, steel: str, against: str,
                    prior: str = ""):
        prompt = (
            f"Hypothesis:\n{hypothesis}\n\n"
            f"Corpus excerpts:\n{context}\n\n"
            f"SUPPORTING argument:\n{steel}\n\n"
            f"OPPOSING argument:\n{against}\n\n"
            + (f"Prior to apply: {prior}\n\n" if prior else "")
            + "Render your verdict now."
        )
        raw = self.judge_llm.complete(_JUDGE_SYSTEM, prompt, max_tokens=400)
        return _parse_verdict(raw)


def _parse_verdict(raw: str) -> tuple[str, str]:
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        try:
            d = json.loads(raw[start:end + 1])
            v = str(d.get("verdict", "")).strip().lower()
            if v in {"supported", "neutral", "contradicted"}:
                return v, str(d.get("reason", ""))
        except json.JSONDecodeError:
            pass
    # If the judge is unparseable, fall back to NEUTRAL (unverified) rather than refuting —
    # never silently kill a claim on a parse error.
    return "neutral", "unparseable adjudication - defaulting to neutral (unverified)"


# --- Atomic-fact decomposition falsifier (FActScore / Min et al. 2023) ---
#
# The debate panel recovered recall but lost precision: judging a COMPOUND claim holistically, the
# model anchored on the mostly-correct parts and missed the one wrong atom (a substitution like
# glaucoma -> hypertension). The fix is to decompose the claim into atomic facts, classify EACH
# against the corpus (supported | neutral | contradicted), and refute if ANY atom is contradicted.

_DECOMPOSE_SYSTEM = (
    "Break the hypothesis into its atomic factual claims - each a single, independently checkable "
    "assertion (one subject, one predicate). Do not add facts not present in the hypothesis. "
    "Return a JSON array of short strings."
)
_ATOM_JUDGE_SYSTEM = (
    "You classify each atomic claim against the corpus excerpts as exactly one of "
    "supported | neutral | contradicted.\n"
    "  supported   - the corpus entails the atom (faithful paraphrases count).\n"
    "  contradicted- CLOSED-WORLD ON STATED ATTRIBUTES: treat the corpus as the complete record "
    "for any attribute it states. If the corpus gives a value for an attribute, an atom asserting "
    "a DIFFERENT value for that same attribute is contradicted - you do NOT need an explicit 'not "
    "Y' (corpus 'GC-MS' vs atom 'NMR'; corpus 'glaucoma' vs atom 'hypertension'; corpus '6.33' vs "
    "atom 'above 8'; corpus 'three agents' vs atom 'two'). Negating a stated fact is also "
    "contradicted.\n"
    "  neutral     - the corpus states nothing about that attribute at all.\n"
    "Do not use outside knowledge. "
    'Return a JSON array: [{"atom": "...", "verdict": "supported|neutral|contradicted"}].'
)


class DecompositionFalsifier:
    """Decompose into atoms, judge each, refute if any atom is contradicted. Same interface."""

    def __init__(self, llm: LLM, judge_llm: LLM | None = None):
        self.llm = llm
        self.judge_llm = judge_llm or llm
        self.n_votes = 1

    def attempt(self, hypothesis: str, corpus: Corpus) -> Refutation:
        hits = corpus.keyword_search(hypothesis, k=8)
        grounding = [h.chunk_id for h in hits]
        context = "\n\n".join(f"[{h.chunk_id}] {h.text[:700]}" for h in hits)

        atoms = self._decompose(hypothesis)
        judged = self._judge_atoms(atoms, context)  # list[(atom, verdict)]

        verdicts = [v for _, v in judged]
        if "contradicted" in verdicts:
            overall = "contradicted"
        elif verdicts and all(v == "supported" for v in verdicts):
            overall = "supported"
        else:
            overall = "neutral"
        bad = [a for a, v in judged if v == "contradicted"]
        reason = ("contradicted atoms: " + "; ".join(bad)) if bad else f"no atom contradicted ({overall})"
        return Refutation(hypothesis=hypothesis, refuted=(overall == "contradicted"),
                          reason=reason, grounding=grounding, verdict=overall)

    def _decompose(self, hypothesis: str) -> list[str]:
        raw = self.llm.complete(_DECOMPOSE_SYSTEM, f"Hypothesis:\n{hypothesis}", max_tokens=400)
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            try:
                items = json.loads(raw[start:end + 1])
                atoms = [str(x).strip() for x in items if str(x).strip()]
                if atoms:
                    return atoms
            except json.JSONDecodeError:
                pass
        return [hypothesis]  # fall back to judging the whole claim

    def _judge_atoms(self, atoms: list[str], context: str) -> list[tuple[str, str]]:
        numbered = "\n".join(f"{i+1}. {a}" for i, a in enumerate(atoms))
        prompt = (f"Corpus excerpts (your ONLY allowed evidence):\n{context}\n\n"
                  f"Atomic claims:\n{numbered}\n\nClassify each now.")
        raw = self.judge_llm.complete(_ATOM_JUDGE_SYSTEM, prompt, max_tokens=700)
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            try:
                items = json.loads(raw[start:end + 1])
                out = []
                for it in items:
                    v = str(it.get("verdict", "")).strip().lower()
                    out.append((str(it.get("atom", "")),
                                v if v in {"supported", "neutral", "contradicted"} else "neutral"))
                if out:
                    return out
            except (json.JSONDecodeError, AttributeError):
                pass
        # Unparseable -> treat every atom as neutral (never refute on a parse error).
        return [(a, "neutral") for a in atoms]
