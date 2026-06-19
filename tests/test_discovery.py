"""Offline tests for the corpus builder and the discovery engine."""

from __future__ import annotations

from poliresearch.corpus import Chunk, Corpus
from poliresearch.corpus_builder import CorpusBuilder, reconstruct_abstract
from poliresearch.discovery import DiscoveryEngine


def test_reconstruct_abstract():
    inv = {"Large": [0], "language": [1], "models": [2], "discover": [3]}
    assert reconstruct_abstract(inv) == "Large language models discover"
    assert reconstruct_abstract(None) == ""


class _Resp:
    def __init__(self, payload):
        self.status_code = 200
        self._p = payload

    def json(self):
        return self._p


class _OneShotSession:
    def __init__(self, works):
        self.works = works
        self.calls = 0

    def get(self, url, params=None, timeout=0):
        self.calls += 1
        # first page returns works + no next cursor -> loop ends
        return _Resp({"results": self.works, "meta": {"next_cursor": None}})


def test_corpus_builder_writes_papers(tmp_path):
    works = [
        {"id": "https://openalex.org/W1", "doi": "https://doi.org/10.1/a", "title": "Paper A",
         "publication_year": 2023, "authorships": [{"author": {"display_name": "Jane Roe"}}],
         "abstract_inverted_index": {"Hello": [0], "world": [1]}},
        {"id": "https://openalex.org/W2", "title": "No abstract paper", "publication_year": 2022,
         "authorships": [], "abstract_inverted_index": None},
    ]
    b = CorpusBuilder(session=_OneShotSession(works))
    res = b.build("test topic", tmp_path, max_papers=10)
    assert res.papers == 1  # the no-abstract paper is skipped
    files = list(tmp_path.glob("*.txt"))
    assert len(files) == 1 and "Hello world" in files[0].read_text(encoding="utf-8")
    assert res.references[0].doi == "10.1/a"


# --- Discovery engine with a scripted LLM ---
class _ScriptLLM:
    """Generator returns a JSON list; falsifier judge returns supported/contradicted by keyword."""

    def __init__(self):
        self.model = "fake"
        self.available = True

    def complete(self, system, prompt, *, max_tokens=1500):
        if "hypothesis generator" in system:
            return '["ABCA1 links ROCK inhibition to phagocytosis", "Bad claim about nothing"]'
        if "adjudicator" in system:
            # contradict the "Bad claim" (dropped); the ABCA1 synthesis is novel -> neutral lead
            v = "contradicted" if "Bad claim" in prompt else "neutral"
            return '{"verdict": "%s", "reason": "x"}' % v
        if "advocate" in system or "skeptic" in system:
            return "argument"
        return "synthesis report text"


def _corpus():
    return Corpus(root=".", chunks=[
        Chunk("p.txt#0", "p.txt", "ROCK inhibition increases RPE phagocytosis via ABCA1."),
    ], corpus_hash="h")


def test_discovery_engine_produces_candidate_discovery(tmp_path):
    eng = DiscoveryEngine(_ScriptLLM())
    res = eng.run("link ROCK inhibition to phagocytosis", _corpus(),
                  cycles=1, rollouts=2, runs_dir=str(tmp_path))
    # the ABCA1 hypothesis is SUPPORTED -> candidate discovery; the bad one is contradicted/dropped
    assert len(res.candidate_discoveries) == 1
    assert "ABCA1" in res.candidate_discoveries[0].text
    assert res.report  # synthesis produced
