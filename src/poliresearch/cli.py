"""Command-line interface.

    python -m poliresearch verify-doi 10.1038/s41586-023-06792-0
    python -m poliresearch check-bibliography examples/sample_bibliography.json
    python -m poliresearch gates examples/sample_claims.json
    python -m poliresearch ingest --corpus ./corpus
    python -m poliresearch ask "your question" --corpus ./corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .checklist import format_verdict, run_checklist
from .config import load_settings
from .corpus import load_corpus
from .domain import get_domain
from .models import Claim, Reference
from .reference_verification import UnifiedVerifier


def _print_check(label_id: str, chk) -> None:
    print(f"Identifier: {label_id}  (source: {chk.source})")
    print(f"  exists       : {chk.exists}")
    print(f"  retracted    : {chk.retracted}")
    print(f"  authors_match: {chk.authors_match}")
    print(f"  year_match   : {chk.year_match}")
    print(f"  title_match  : {chk.title_match}")
    if chk.title:
        print(f"  title        : {chk.title}")
    if chk.error:
        print(f"  note         : {chk.error}")
    print(f"  => {'TRUSTWORTHY' if chk.ok else 'REJECTED (gate fail)'}")


def _cmd_verify_doi(args, settings) -> int:
    chk = UnifiedVerifier(mailto=settings.crossref_mailto).verify(Reference(doi=args.doi))
    _print_check(args.doi, chk)
    return 0 if chk.ok else 1


def _cmd_verify_arxiv(args, settings) -> int:
    chk = UnifiedVerifier(mailto=settings.crossref_mailto).verify(Reference(arxiv_id=args.arxiv_id))
    _print_check(f"arXiv:{args.arxiv_id}", chk)
    return 0 if chk.ok else 1


def _cmd_verify_pmid(args, settings) -> int:
    chk = UnifiedVerifier(mailto=settings.crossref_mailto).verify(Reference(pmid=args.pmid))
    _print_check(f"PMID:{args.pmid}", chk)
    return 0 if chk.ok else 1


def _cmd_verify_title(args, settings) -> int:
    chk = UnifiedVerifier(mailto=settings.crossref_mailto).verify(Reference(title=args.title))
    _print_check(f"title:{args.title}", chk)
    return 0 if chk.ok else 1


def _cmd_check_bibliography(args, settings) -> int:
    refs = [Reference(**r) for r in json.loads(Path(args.file).read_text(encoding='utf-8'))]
    v = UnifiedVerifier(mailto=settings.crossref_mailto)
    bad = 0
    for ref, chk in zip(refs, v.verify_many(refs)):
        flag = "OK " if chk.ok else "BAD"
        if not chk.ok:
            bad += 1
        label = chk.title or ref.title or "?"
        ident = ref.identifier or "(no id)"
        print(f"[{flag}] {ident} - {label}"
              + (f"  <{chk.error}>" if chk.error else "")
              + ("  <RETRACTED>" if chk.retracted else ""))
    print(f"\n{len(refs) - bad}/{len(refs)} references trustworthy.")
    return 0 if bad == 0 else 1


def _cmd_gates(args, settings) -> int:
    claims = [Claim.from_dict(d) for d in json.loads(Path(args.file).read_text(encoding='utf-8'))]
    v = UnifiedVerifier(mailto=settings.crossref_mailto)
    chunk_ids = set()
    corpus_hash = None
    if args.corpus and Path(args.corpus).exists():
        c = load_corpus(args.corpus)
        chunk_ids, corpus_hash = c.chunk_ids(), c.corpus_hash
    n_acc = 0
    for claim in claims:
        verdict = run_checklist(claim, verifier=v, corpus_chunk_ids=chunk_ids,
                                corpus_hash=corpus_hash,
                                falsification_attempted=False, human_reviewed=False)
        print(format_verdict(claim, verdict))
        print()
        n_acc += int(verdict.accepted)
    print(f"{n_acc}/{len(claims)} claims fully accepted "
          f"(others await falsification + human sign-off — gates 7-8).")
    return 0


def _cmd_build_corpus(args, settings) -> int:
    """Fetch real papers for a topic from OpenAlex into a corpus dir (scale, no key)."""
    from .corpus_builder import CorpusBuilder
    out = args.out or str(settings.corpus_dir)
    res = CorpusBuilder(mailto=settings.crossref_mailto).build(
        args.topic, out, max_papers=args.papers, fulltext=args.fulltext)
    extra = " + OA full text" if args.fulltext else ""
    print(f"Built corpus '{args.topic}': {res.papers} papers (abstracts{extra}) -> {res.out_dir}")
    print(f"Now run:  python -m poliresearch discover \"{args.topic}\" --corpus {res.out_dir}")
    return 0 if res.papers else 1


def _cmd_discover(args, settings) -> int:
    """Kosmos-style iterative parallel discovery loop over a corpus (needs LLM)."""
    from .llm import make_llm, resolve_backend
    from .discovery import DiscoveryEngine

    llm = make_llm(settings)
    if not llm.available:
        print("Discovery needs an LLM. Install the Claude Code CLI (no key) or set "
              "ANTHROPIC_API_KEY.", file=sys.stderr)
        return 2
    corpus = load_corpus(args.corpus or str(settings.corpus_dir))
    falsifier_llm = make_llm(settings, model=settings.falsifier_model)
    engine = DiscoveryEngine(llm, mailto=settings.crossref_mailto, falsifier_llm=falsifier_llm)
    print(f"Discovery: backend={resolve_backend(settings)}  corpus={len(corpus.chunks)} chunks  "
          f"cycles={args.cycles} x rollouts={args.rollouts}\n")
    res = engine.run(args.goal, corpus, cycles=args.cycles, rollouts=args.rollouts,
                     runs_dir=str(settings.runs_dir))

    print(f"=== {len(res.candidate_discoveries)} NOVEL candidate discoveries "
          f"(survived falsification; beyond the corpus -> to be tested) ===")
    for d in res.candidate_discoveries:
        print(f"  [c{d.cycle}] {d.text}")
    print(f"\n=== {len(res.corroborated)} corroborated claims "
          f"(SUPPORTED by the corpus; verification wins, not novel) ===")
    for c in res.corroborated[:10]:
        print(f"  [c{c.cycle}] {c.text}")
    print("\n=== synthesis ===\n" + res.report)
    print("\nNOTE: novel candidates are falsifiable leads requiring experimental/human "
          "validation (as Kosmos itself requires). For CS/AI, the experiment loop can test them.")
    return 0


def _cmd_obsidian(args, settings) -> int:
    """Export the corpus as an Obsidian vault (paper-to-paper links for the graph view)."""
    from .obsidian import export_vault
    corpus_dir = args.corpus or str(settings.corpus_dir)
    n = export_vault(corpus_dir, args.out, top_k=args.top_k)
    print(f"Wrote {n} paper notes to '{args.out}'.")
    print(f"Open '{args.out}' as an Obsidian vault and use Graph view to see citation / "
          f"shared-author / similarity links.")
    return 0


def _cmd_test_hypothesis(args, settings) -> int:
    """CS/AI experiment loop: the Experimenter writes a self-contained test, the runner runs it.
    A PASS is empirical support *under the experiment's own assumptions* — not proof; real-model
    claims need a real model (torch/transformers) the runner may not have."""
    from .llm import make_llm
    from .agents.experimenter import Experimenter
    from .experiment import ExperimentRunner
    from .domain import get_domain

    llm = make_llm(settings)
    if not llm.available:
        print("Needs an LLM (Claude Code CLI or ANTHROPIC_API_KEY).", file=sys.stderr)
        return 2
    framing = get_domain(args.domain or settings.domain_key).prompt_framing
    code = Experimenter(llm).write_experiment(args.hypothesis, framing=framing)
    if not code:
        print("No experiment code was generated.")
        return 1
    print("=== generated experiment ===\n" + code + "\n")
    res = ExperimentRunner(timeout_s=args.timeout).run_python(code)
    print("=== result: " + res.verdict_line + " ===")
    if res.stdout:
        print(res.stdout)
    if res.stderr and not res.success:
        print("stderr:\n" + res.stderr[-1500:])
    return 0 if res.success else 1


def _cmd_sources(args, settings) -> int:
    """List keyed paper sources and whether each is enabled (key set) or disabled."""
    from .sources import REGISTRY
    print("Keyless anchors (always on): Crossref, arXiv, PubMed, OpenAlex\n")
    print("Keyed sources (disabled by default unless the API key env var is set):")
    for s in REGISTRY:
        print(f"  [{'x' if s.enabled else ' '}] {s.name:18} {s.env_var:26} {s.status}")
        print(f"       {s.description}")
    return 0


def _cmd_ingest(args, settings) -> int:
    corpus = load_corpus(args.corpus or settings.corpus_dir)
    print(f"Indexed {len(corpus.chunks)} chunks from "
          f"{len({c.source for c in corpus.chunks})} files.")
    print(f"Corpus hash (gate 6 / reproducibility): {corpus.corpus_hash}")
    return 0


def _cmd_experiment(args, settings) -> int:
    """Run a Python experiment file through the sandboxed runner (no LLM needed)."""
    from .experiment import ExperimentRunner
    code = Path(args.file).read_text(encoding="utf-8")
    res = ExperimentRunner(timeout_s=args.timeout).run_python(code)
    print(res.verdict_line)
    if res.stdout:
        print("--- stdout ---\n" + res.stdout)
    if res.stderr:
        print("--- stderr ---\n" + res.stderr)
    return 0 if res.success else 1


def _cmd_evaluate(args, settings) -> int:
    from .evaluation import dataset as ds
    from .evaluation import (evaluate_citations, evaluate_experiments, compare_falsifier)
    from .experiment import ExperimentRunner
    from .reference_verification import UnifiedVerifier

    suffix = "_adversarial" if args.suite == "adversarial" else ""
    basename = {"falsifier": "falsification"}.get(args.kind, args.kind)  # file != cmd name
    default = f"eval/datasets/{basename}{suffix}.json"
    path = args.dataset or default

    if args.kind == "citations":
        report = evaluate_citations(UnifiedVerifier(mailto=settings.crossref_mailto),
                                    ds.load_citations(path))
        print(report.fmt(verbose=args.verbose))
        return 0

    if args.kind == "experiments":
        report = evaluate_experiments(ExperimentRunner(timeout_s=args.timeout),
                                      ds.load_experiments(path))
        print(report.fmt(verbose=args.verbose))
        return 0

    if args.kind == "falsifier":
        votes = [int(v) for v in args.votes.split(",")]
        examples = ds.load_falsification(path)

        # Simulation mode: measure the vote-aggregation effect with no LLM/key.
        if args.simulate is not None:
            from .evaluation import simulate_falsifier
            metrics = simulate_falsifier(examples, votes, reliability=args.simulate,
                                         trials=args.trials)
            print(f"Vote-aggregation simulation on {len(examples)} adversarial examples "
                  f"(single-judge reliability p={args.simulate}, {args.trials} trials/example):\n")
            base = metrics[votes[0]].accuracy
            for v in votes:
                acc = metrics[v].accuracy
                delta = f"  (+{acc - base:.3f} vs {votes[0]}-vote)" if v != votes[0] else ""
                print(f"  n_votes={v}: accuracy={acc:.3f}  f1={metrics[v].f1:.3f}{delta}")
            best = max(votes, key=lambda v: metrics[v].accuracy)
            print(f"\n=> best accuracy at n_votes={best}. Majority voting helps whenever the "
                  f"single-judge reliability exceeds 0.5 (variance reduction).")
            return 0

        from .llm import make_llm, resolve_backend
        if not make_llm(settings).available:
            print("Falsifier evaluation needs an LLM: install the Claude Code CLI (no key) or "
                  "set ANTHROPIC_API_KEY, or use --simulate P (e.g. 0.75) to measure the\n"
                  "vote-aggregation effect with neither. "
                  "(`evaluate citations`/`experiments` also run keyless.)",
                  file=sys.stderr)
            return 2
        print(f"(LLM backend: {resolve_backend(settings)})")
        corpus = load_corpus(args.corpus or str(settings.corpus_dir))

        # Debate / decompose modes: the literature-backed fixes.
        if args.mode in ("debate", "decompose"):
            from .evaluation import evaluate_falsifier, three_action_from_report
            rep = evaluate_falsifier(make_llm(settings, model=settings.falsifier_model),
                                     corpus, examples, mode=args.mode)
            label = ("Debate panel (steelman/refuter/adjudicator)" if args.mode == "debate"
                     else "Atomic decomposition (refute if any atom contradicted)")
            print(f"{label}, refute only on CONTRADICTED\n")
            print(rep.fmt(verbose=args.verbose))
            print("\n-- three-action scoring (supported=accept, contradicted=reject, "
                  "neutral=flag/defer) --")
            print(three_action_from_report(rep).fmt())
            return 0

        reports = compare_falsifier(lambda: make_llm(settings, model=settings.falsifier_model),
                                    corpus, examples, votes)
        print("Comparing falsifier vote counts (does independent 3-vote beat 1-vote?)\n")
        for v in votes:
            print(reports[v].fmt(verbose=args.verbose))
            print()
        best = max(votes, key=lambda v: reports[v].metrics.f1)
        print(f"=> best F1 at n_votes={best} "
              f"(F1={reports[best].metrics.f1:.3f})")
        return 0

    print(f"unknown evaluation kind: {args.kind}", file=sys.stderr)
    return 2


def _cmd_ask(args, settings) -> int:
    from .llm import make_llm
    from .pipeline import Pipeline

    from .llm import resolve_backend
    domain = get_domain(args.domain or settings.domain_key)
    llm = make_llm(settings)
    if not llm.available:
        print("No LLM available. Install the Claude Code CLI (no key needed) or set "
              "ANTHROPIC_API_KEY in .env to run `ask`.\n"
              "The verification commands (verify-doi, verify-arxiv, check-bibliography, gates, "
              "ingest, experiment) work without any LLM.", file=sys.stderr)
        return 2
    print(f"LLM backend: {resolve_backend(settings)}")

    falsifier_llm = make_llm(settings, model=settings.falsifier_model)  # independent judge
    corpus_dir = args.corpus or str(settings.corpus_dir)
    pipe = Pipeline(llm, mailto=settings.crossref_mailto, runs_dir=str(settings.runs_dir),
                    falsifier_llm=falsifier_llm, n_votes=args.votes, domain=domain,
                    falsifier_mode=settings.falsifier_mode)
    print(f"Domain: {domain.name}  |  experiments: {domain.enable_experiments}  |  "
          f"falsifier: {settings.falsifier_mode}")
    print(f"Generator: {settings.model}  |  Falsifier: {settings.falsifier_model} "
          f"x{args.votes} votes\n")
    result = pipe.run(args.question, corpus_dir, n=args.n)

    print(f"Question: {result.question}")
    print(f"Corpus hash: {result.corpus_hash}\n")
    print(f"{len(result.survivors)}/{len(result.results)} hypotheses survived falsification:\n")
    for r in result.results:
        head = "SURVIVED" if not r.refuted else "REFUTED "
        print(f"[{head}] {r.text}")
        if r.refuted:
            print(f"           reason: {r.reason}")
        elif r.verdict:
            pend = ", ".join(g.name for g in r.verdict.pending_human)
            print(f"           mechanical gates: "
                  f"{'all pass' if not r.verdict.failed_gates else 'FAIL'}; "
                  f"awaiting human: {pend or 'none'}")
        if r.experiment:
            print(f"           empirical test: {r.experiment.verdict_line}")
        print()
    print(f"World model + provenance written under: {settings.runs_dir}")
    print("NOTE: no hypothesis is publishable until gates 7-8 (human) are signed off.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="poliresearch",
                                description="Falsification-first, verification-gated AI research system.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("verify-doi", help="Verify one DOI (Crossref + Retraction Watch).")
    s.add_argument("doi")
    s.set_defaults(func=_cmd_verify_doi)

    s = sub.add_parser("verify-arxiv", help="Verify one arXiv id (e.g. 2408.06292).")
    s.add_argument("arxiv_id")
    s.set_defaults(func=_cmd_verify_arxiv)

    s = sub.add_parser("verify-pmid", help="Verify one PubMed id (biomedical).")
    s.add_argument("pmid")
    s.set_defaults(func=_cmd_verify_pmid)

    s = sub.add_parser("verify-title", help="Verify a paper by title via OpenAlex (no id needed).")
    s.add_argument("title")
    s.set_defaults(func=_cmd_verify_title)

    s = sub.add_parser("experiment", help="Run a Python experiment file in the sandboxed runner.")
    s.add_argument("file")
    s.add_argument("--timeout", type=int, default=30)
    s.set_defaults(func=_cmd_experiment)

    s = sub.add_parser("check-bibliography", help="Verify a JSON list of references.")
    s.add_argument("file")
    s.set_defaults(func=_cmd_check_bibliography)

    s = sub.add_parser("gates", help="Run the 8-gate anti-hallucination checklist on claims.")
    s.add_argument("file")
    s.add_argument("--corpus", default=None)
    s.set_defaults(func=_cmd_gates)

    s = sub.add_parser("ingest", help="Index a closed corpus and print its hash.")
    s.add_argument("--corpus", default=None)
    s.set_defaults(func=_cmd_ingest)

    s = sub.add_parser("sources", help="List keyed paper sources and their enabled/disabled state.")
    s.set_defaults(func=_cmd_sources)

    s = sub.add_parser("test-hypothesis", help="Empirically test a hypothesis via the code-experiment loop.")
    s.add_argument("hypothesis")
    s.add_argument("--domain", default=None)
    s.add_argument("--timeout", type=int, default=60)
    s.set_defaults(func=_cmd_test_hypothesis)

    s = sub.add_parser("obsidian", help="Export the corpus as an Obsidian vault (graph of links).")
    s.add_argument("--corpus", default=None)
    s.add_argument("--out", default="./vault")
    s.add_argument("--top-k", type=int, default=5, help="similarity links per paper")
    s.set_defaults(func=_cmd_obsidian)

    s = sub.add_parser("build-corpus", help="Fetch N real papers for a topic from OpenAlex (scale).")
    s.add_argument("topic")
    s.add_argument("--out", default=None)
    s.add_argument("--papers", type=int, default=2000)
    s.add_argument("--fulltext", action="store_true",
                   help="also fetch open-access full text (arXiv/Unpaywall) where available")
    s.set_defaults(func=_cmd_build_corpus)

    s = sub.add_parser("discover", help="Kosmos-style iterative parallel discovery loop (needs LLM).")
    s.add_argument("goal")
    s.add_argument("--corpus", default=None)
    s.add_argument("--cycles", type=int, default=3)
    s.add_argument("--rollouts", type=int, default=6, help="parallel hypotheses per cycle")
    s.set_defaults(func=_cmd_discover)

    s = sub.add_parser("evaluate", help="Measure the system against labeled datasets.")
    s.add_argument("kind", choices=["citations", "experiments", "falsifier"])
    s.add_argument("--suite", choices=["base", "adversarial"], default="base",
                   help="base (easy) or adversarial (paraphrases, subtle errors, near-misses)")
    s.add_argument("--dataset", default=None, help="override dataset path")
    s.add_argument("--corpus", default=None, help="(falsifier) corpus dir")
    s.add_argument("--votes", default="1,3", help="(falsifier) comma-separated vote counts")
    s.add_argument("--simulate", type=float, default=None,
                   help="(falsifier) measure vote-aggregation with no LLM; value = single-judge "
                        "reliability p in [0,1], e.g. 0.75")
    s.add_argument("--trials", type=int, default=4000, help="(falsifier --simulate) trials/example")
    s.add_argument("--mode", choices=["vote", "debate", "decompose"], default="vote",
                   help="(falsifier) vote=skeptic majority; debate=steelman/refuter/judge; "
                        "decompose=atomic-fact decomposition")
    s.add_argument("--timeout", type=int, default=30, help="(experiments) per-script timeout s")
    s.add_argument("--verbose", action="store_true", help="show per-example results")
    s.set_defaults(func=_cmd_evaluate)

    s = sub.add_parser("ask", help="Run the full falsification-first pipeline (needs LLM).")
    s.add_argument("question")
    s.add_argument("--corpus", default=None)
    s.add_argument("-n", type=int, default=4, help="number of hypotheses to generate")
    s.add_argument("--votes", type=int, default=3,
                   help="independent falsification votes per hypothesis (majority refutes)")
    s.add_argument("--domain", default=None, help="domain profile: cs_ai (default) | generic | biomed")
    s.set_defaults(func=_cmd_ask)
    return p


def main(argv: list[str] | None = None) -> int:
    # Force UTF-8 console output: results contain Unicode (≤, ×, em-dashes) that the Windows
    # cp1252 codec cannot encode, which otherwise crashes printing mid-output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings()
    return args.func(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
