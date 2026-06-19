"""Export a corpus as an Obsidian vault so paper-to-paper links show in the graph view.

Each paper becomes a markdown note; edges are written as [[wikilinks]] from three signals:
  * Citations    — OpenAlex `referenced_works` that are also in the corpus (the real graph)
  * Shared authors
  * Semantic similarity — top-k nearest papers by BM25 (guarantees a connected graph even when
    citation data is absent, e.g. the curated sample corpus)

Open the output folder as an Obsidian vault and the Graph view renders the network.
"""

from __future__ import annotations

import re
from pathlib import Path

from .corpus import Chunk, load_corpus
from .retrieval import BM25Retriever

_ILLEGAL = re.compile(r'[\\/:*?"<>|#\[\]^]')


def _field(text: str, name: str) -> str | None:
    m = re.search(rf"^{name}:\s*(.*)$", text, re.M)
    return m.group(1).strip() if m else None


def _title_of(text: str, fallback: str) -> str:
    t = _field(text, "Title")
    if t:
        return t
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:120]
    return fallback


def _surnames(authors: list[str]) -> set[str]:
    out = set()
    for a in authors:
        a = a.strip()
        if a:
            out.add((a.split(",")[0] if "," in a else a.split()[-1]).strip().lower())
    return out


def _safe(name: str) -> str:
    name = _ILLEGAL.sub(" ", name)
    return re.sub(r"\s+", " ", name).strip()[:100] or "untitled"


def export_vault(corpus_dir: str | Path, out_dir: str | Path, top_k: int = 5) -> int:
    corpus = load_corpus(corpus_dir)
    papers: dict[str, list[str]] = {}
    for c in corpus.chunks:
        papers.setdefault(c.source, []).append(c.text)
    sources = list(papers)
    text = {s: "\n".join(papers[s]) for s in sources}

    title = {s: _title_of(text[s], s) for s in sources}
    authors = {s: [x.strip() for x in (_field(text[s], "Authors") or "").split(",") if x.strip()]
               for s in sources}
    surn = {s: _surnames(authors[s]) for s in sources}
    oaid = {s: (_field(text[s], "OpenAlexID") or Path(s).stem) for s in sources}
    refs = {s: [r.strip() for r in (_field(text[s], "References") or "").split(";") if r.strip()]
            for s in sources}
    by_oaid = {oaid[s]: s for s in sources}

    # unique, readable note names
    name, used = {}, set()
    for s in sources:
        base = _safe(title[s])
        n, i = base, 2
        while n in used:
            n, i = f"{base} ({i})", i + 1
        used.add(n)
        name[s] = n

    # similarity edges (BM25 over per-paper docs)
    bm = BM25Retriever([Chunk(s, s, text[s]) for s in sources])
    sim = {}
    for s in sources:
        hits = [h.chunk_id for h in bm.search(text[s][:2000], k=top_k + 1) if h.chunk_id != s]
        sim[s] = hits[:top_k]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for s in sources:
        cites = [name[by_oaid[r]] for r in refs[s] if r in by_oaid]
        shared = [name[t] for t in sources if t != s and surn[s] & surn[t]][:10]
        related = [name[t] for t in sim[s]]
        abstract = (_field(text[s], "Abstract") or text[s][:600])
        lines = [
            "---",
            f"title: {title[s]}",
            f"year: {_field(text[s], 'Year') or ''}",
            f"doi: {_field(text[s], 'DOI') or ''}",
            "---",
            f"# {title[s]}",
            "",
            f"**Authors:** {', '.join(authors[s]) or '?'}",
            "",
            (abstract[:800] + ("…" if len(abstract) > 800 else "")),
            "",
        ]
        if cites:
            lines += ["## Cites", " ".join(f"[[{c}]]" for c in cites), ""]
        if shared:
            lines += ["## Shared authors", " ".join(f"[[{c}]]" for c in shared), ""]
        if related:
            lines += ["## Related (similarity)", " ".join(f"[[{c}]]" for c in related), ""]
        (out / f"{name[s]}.md").write_text("\n".join(lines), encoding="utf-8")

    (out / "_index.md").write_text(
        "# Corpus\n\n" + "\n".join(f"- [[{name[s]}]]" for s in sources), encoding="utf-8")
    return len(sources)
