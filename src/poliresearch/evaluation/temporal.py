"""Temporal-rediscovery benchmark (H7) — a manipulation-resistant novelty metric.

Freeze a corpus at year X, run discovery, and measure how many of the held-out findings actually
published in X+1..X+window the system "rediscovered". Unlike expert-surprise ratings (which reward
incoherent far-bridges that never validate), temporal rediscovery is grounded in what was actually
found later. This module provides the deterministic split + scorer; the live measurement is: run
`DiscoveryEngine` on the train split, then `rediscovery_rate(candidates, holdout)`.
"""

from __future__ import annotations

import re
from pathlib import Path

_STOP = {"the", "a", "an", "of", "for", "with", "and", "to", "in", "on", "is", "are", "that",
         "by", "as", "from", "we", "our", "this"}


def _year(text: str) -> int | None:
    m = re.search(r"^Year:\s*(\d{4})", text, re.M)
    return int(m.group(1)) if m else None


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2 and t not in _STOP}


def split_corpus_by_year(corpus_dir: str | Path, cutoff: int, window: int = 3
                         ) -> tuple[list[str], list[str]]:
    """Return (train_texts ≤ cutoff, holdout_texts in cutoff+1..cutoff+window)."""
    train, holdout = [], []
    for path in sorted(Path(corpus_dir).glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        y = _year(text)
        if y is None:
            continue
        if y <= cutoff:
            train.append(text)
        elif cutoff < y <= cutoff + window:
            holdout.append(text)
    return train, holdout


def rediscovery_rate(candidate_texts: list[str], holdout_texts: list[str],
                     threshold: float = 0.30) -> dict:
    """Fraction of held-out findings 'rediscovered' by some candidate (content-token overlap of
    the holdout's tokens covered by a candidate ≥ threshold). Manipulation-resistant: a candidate
    only counts if it lexically matches something actually published later."""
    cand_tok = [_tokens(c) for c in candidate_texts]
    hits = 0
    matched = []
    for h in holdout_texts:
        ht = _tokens(h)
        if not ht:
            continue
        best = max((len(ht & c) / len(ht) for c in cand_tok), default=0.0)
        if best >= threshold:
            hits += 1
            matched.append(round(best, 2))
    n = len([h for h in holdout_texts if _tokens(h)])
    return {"rediscovered": hits, "holdout": n,
            "rate": (hits / n) if n else 0.0, "match_scores": matched}
