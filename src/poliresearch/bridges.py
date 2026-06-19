"""Cross-corpus bridge finding (H3) — steer generation toward high-novelty paper pairs.

The discovery run's own H3 hypothesis: genuine novelty concentrates in hypotheses that *bridge*
two papers sharing no co-authors and no common citations, at an INTERMEDIATE distance — coherent
enough to connect (not ≥7 hops apart) but not redundant (not ≤1 hop). We operationalize the
inverted-U as: no shared authors AND no shared references AND mid-band topical similarity. Feeding
these pairs into the generator biases it toward the productive middle instead of trivial or
incoherent pairings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus

_STOP = {"the", "a", "an", "of", "for", "with", "and", "to", "in", "on", "is", "are", "that",
         "by", "as", "from", "we", "our", "this", "abstract", "title", "year", "authors", "doi"}


def _field(text: str, name: str) -> str:
    m = re.search(rf"^{name}:\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else ""


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) > 2 and t not in _STOP}


@dataclass
class _Paper:
    source: str
    title: str
    tokens: set[str]
    authors: set[str]
    refs: set[str]


def _papers(corpus: Corpus) -> list[_Paper]:
    by_src: dict[str, list[str]] = {}
    for c in corpus.chunks:
        by_src.setdefault(c.source, []).append(c.text)
    out = []
    for src, parts in by_src.items():
        text = "\n".join(parts)
        title = _field(text, "Title") or src
        authors = {a.split(",")[0].strip().lower() if "," in a else a.split()[-1].strip().lower()
                   for a in _field(text, "Authors").split(",") if a.strip()}
        refs = {r.strip() for r in _field(text, "References").split(";") if r.strip()}
        out.append(_Paper(src, title, _tokens(text), authors, refs))
    return out


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bridge_pairs(corpus: Corpus, k: int = 6, low: float = 0.08, high: float = 0.35
                 ) -> list[tuple[str, str, float]]:
    """Return up to k (title_a, title_b, similarity) bridges: no shared authors, no shared
    references, and mid-band topical similarity (the inverted-U's productive middle)."""
    papers = _papers(corpus)
    cands = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            p, q = papers[i], papers[j]
            if p.authors & q.authors:          # co-authors -> too near
                continue
            if p.refs & q.refs:                # common citations -> too near
                continue
            sim = _jaccard(p.tokens, q.tokens)
            if low < sim < high:               # coherent but not redundant
                cands.append((p.title, q.title, round(sim, 3)))
    # rank toward the middle of the band (most "bridge-like"), then dedupe titles lightly
    mid = (low + high) / 2
    cands.sort(key=lambda t: abs(t[2] - mid))
    return cands[:k]


def format_bridges(pairs: list[tuple[str, str, float]]) -> str:
    if not pairs:
        return ""
    lines = "\n".join(f'- "{a}"  <->  "{b}"' for a, b, _ in pairs)
    return ("High-yield cross-paper BRIDGES (these pairs share no authors and no citations, yet "
            "are topically adjacent — the productive middle for novel synthesis). Prefer "
            "hypotheses that connect such distant-but-coherent pairs:\n" + lines)


def bridge_framing(corpus: Corpus, k: int = 6) -> str:
    return format_bridges(bridge_pairs(corpus, k=k))


def bridge_distance_profile(corpus: Corpus, bins: int = 5) -> list[tuple[float, float, int]]:
    """N4: distribution of no-shared-author/no-shared-ref pair similarities across distance bins.
    The similarity axis is the (inverse) bridge-distance; this lets you see where pairs sit and
    tune the band toward the productive intermediate region rather than guessing."""
    papers = _papers(corpus)
    sims = []
    for i in range(len(papers)):
        for j in range(i + 1, len(papers)):
            p, q = papers[i], papers[j]
            if p.authors & q.authors or p.refs & q.refs:
                continue
            s = _jaccard(p.tokens, q.tokens)
            if s > 0:
                sims.append(s)
    out = []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        out.append((round(lo, 2), round(hi, 2), sum(1 for s in sims if lo <= s < hi)))
    return out
