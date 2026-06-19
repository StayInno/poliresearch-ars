"""Offline tests for corpus indexing/hashing and the world model."""

from __future__ import annotations

from poliresearch.corpus import load_corpus
from poliresearch.world_model import WorldModel


def _write_corpus(tmp_path):
    (tmp_path / "a.txt").write_text("ROCK inhibitor ripasudil increases RPE phagocytosis via ABCA1.",
                                    encoding="utf-8")
    (tmp_path / "b.md").write_text("KIRA6 is an IRE1-alpha inhibitor studied in AML.",
                                   encoding="utf-8")
    return tmp_path


def test_corpus_hash_is_deterministic_and_sensitive(tmp_path):
    d = _write_corpus(tmp_path)
    h1 = load_corpus(d).corpus_hash
    h2 = load_corpus(d).corpus_hash
    assert h1 == h2 and h1  # deterministic, non-empty

    (d / "a.txt").write_text("changed content", encoding="utf-8")
    assert load_corpus(d).corpus_hash != h1  # sensitive to change (gate 6)


def test_keyword_search_finds_relevant_chunk(tmp_path):
    c = load_corpus(_write_corpus(tmp_path))
    hits = c.keyword_search("ripasudil RPE phagocytosis", k=1)
    assert hits and "ripasudil" in hits[0].text.lower()


def test_world_model_roundtrip(tmp_path):
    wm = WorldModel(goal="test", corpus_hash="abc")
    i = wm.add_claim("claim one", "synthesis")
    wm.set_claim_status(i, "accepted")
    wm.upsert_entity("ripasudil", kind="drug", target="ROCK")
    wm.add_open_question("does it work in vivo?")
    path = tmp_path / "wm.json"
    wm.save(path)

    loaded = WorldModel.load(path)
    assert loaded.goal == "test"
    assert loaded.accepted_claims()[0]["text"] == "claim one"
    assert loaded.entities["ripasudil"]["target"] == "ROCK"
    assert "does it work in vivo?" in loaded.open_questions
