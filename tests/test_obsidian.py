"""Offline test for the Obsidian vault export (citation / author / similarity links)."""

from __future__ import annotations

from poliresearch.obsidian import export_vault


def _write_corpus(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "W1.txt").write_text(
        "Title: Autonomous chemical research with LLMs\nYear: 2023\nAuthors: Boiko, Gomes\n"
        "DOI: 10.1/a\nOpenAlexID: W1\nReferences: W2\n\n"
        "Abstract: A GPT-4 planner runs chemistry experiments on a robot.\n", encoding="utf-8")
    (tmp_path / "W2.txt").write_text(
        "Title: Robotic chemistry automation\nYear: 2021\nAuthors: Gomes, Smith\n"
        "DOI: 10.1/b\nOpenAlexID: W2\nReferences: \n\n"
        "Abstract: Automated robotic chemistry and reaction optimization.\n", encoding="utf-8")
    return tmp_path


def test_export_vault_creates_linked_notes(tmp_path):
    corpus = _write_corpus(tmp_path / "corpus")
    vault = tmp_path / "vault"
    n = export_vault(corpus, vault, top_k=3)
    assert n == 2
    notes = {p.name for p in vault.glob("*.md")}
    assert "_index.md" in notes and len(notes) == 3  # 2 papers + index

    paper1 = next(vault.glob("Autonomous*.md")).read_text(encoding="utf-8")
    # citation edge W1 -> W2 (W2 is in-corpus) renders as a wikilink to its title note
    assert "## Cites" in paper1 and "[[Robotic chemistry automation]]" in paper1
    # shared author (Gomes) link present
    assert "## Shared authors" in paper1


def test_export_vault_similarity_only_when_no_citations(tmp_path):
    # curated-style docs with no References line still get similarity links -> connected graph
    c = tmp_path / "corpus"
    c.mkdir()
    (c / "p1.txt").write_text("Falsification and verification in language model agents.", encoding="utf-8")
    (c / "p2.txt").write_text("Verification of language model agents via falsification.", encoding="utf-8")
    vault = tmp_path / "v"
    export_vault(c, vault, top_k=2)
    txt = next(vault.glob("Falsification*.md")).read_text(encoding="utf-8")
    assert "## Related (similarity)" in txt and "[[" in txt
