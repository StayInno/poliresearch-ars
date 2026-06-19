"""Offline demo of the verification layer — runs without an API key (needs network for Crossref).

    python examples/demo.py
"""

from __future__ import annotations

from poliresearch.checklist import format_verdict, run_checklist
from poliresearch.corpus import load_corpus
from poliresearch.models import Claim, ClaimType, Reference


def main():
    print("=== 1. Corpus indexing + hash (gate 6 / reproducibility) ===")
    corpus = load_corpus("corpus")
    print(f"  {len(corpus.chunks)} chunks, hash={corpus.corpus_hash}\n")

    print("=== 2. A well-grounded, well-cited claim through all 8 gates ===")
    good = Claim(
        text="Coscientist executed Pd-catalysed cross-couplings verified by GC-MS.",
        claim_type=ClaimType.CITATION,
        references=[Reference(doi="10.1038/s41586-023-06792-0",
                              authors=["Gomes, G."], year=2023)],
        grounding=[corpus.chunks[0].chunk_id],
    )
    verdict = run_checklist(good, corpus_chunk_ids=corpus.chunk_ids(),
                            corpus_hash=corpus.corpus_hash,
                            falsification_attempted=True, falsification_survived=True,
                            human_reviewed=False)
    print(format_verdict(good, verdict))
    print("  (mechanical gates pass; gates 7-8 await the human — exactly the design intent)\n")

    print("=== 3. A fabricated, ungrounded claim is caught ===")
    bad = Claim(text="An unsupported assertion with no citation.",
                claim_type=ClaimType.SYNTHESIS, references=[], grounding=[])
    verdict = run_checklist(bad, corpus_chunk_ids=corpus.chunk_ids(),
                            corpus_hash=corpus.corpus_hash, human_reviewed=True)
    print(format_verdict(bad, verdict))


if __name__ == "__main__":
    main()
